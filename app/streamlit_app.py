# -*- coding: utf-8 -*-
# ===== streamlit_app.py — v8.5 (La Liga aliases + matcher ES, fallback neopromosse) =====
# - Calendari: API-FOOTBALL (primaria) • fallback: football-data.org • FPL per Premier.
# - Metriche: stagione scorsa football-data.co.uk (+ opzionale stagione corrente 30%).
# - Meteo: Open-Meteo (gratis).
# - Matching robusto: alias estesi (Atlético, Barça, Espanyol, Celta, Mallorca, Elche, Levante, Oviedo, Sociedad…),
#   normalizzazione (rimozione numeri e prefissi, token ES come RC/RCD/CF/UD/“club”/“de”/“futbol”).
# - Fallback automatico per squadre senza storico (usa medie di lega).

import os, sys, io, json, re, time, datetime, unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import streamlit as st
import yaml
from math import exp, lgamma, log

# ----- repo root (se mai servisse pipeline) -----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ====== Tentativo import pipeline (opzionale) ======
HAVE_PIPELINE = True
try:
    from pipeline.modeling.prob_model import combine_strengths, finalize_probability, blended_var_factor
    from pipeline.utils.auto_weather import fetch_openmeteo_conditions, LEAGUE_TZ
    from pipeline.utils.geocode import geocode_team_fallback
except Exception:
    HAVE_PIPELINE = False

    LEAGUE_TZ = {"I1":"Europe/Rome","E0":"Europe/London","SP1":"Europe/Madrid","D1":"Europe/Berlin"}

    def geocode_team_fallback(name, code, autosave=True):
        try:
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": name, "count": 1, "language": "en"},
                             timeout=20)
            js = r.json()
            if js and js.get("results"):
                it = js["results"][0]
                return {"lat": float(it["latitude"]), "lon": float(it["longitude"]), "name": it.get("name")}
        except Exception:
            pass
        return {"lat": None, "lon": None, "name": None}

    def fetch_openmeteo_conditions(lat, lon, date_iso, hour_local=18, tz="Europe/Rome"):
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "forecast_days": 7,
                "timezone": "auto"
            }
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
            js = r.json()
            hh = js.get("hourly", {})
            times = hh.get("time", [])
            temps = hh.get("temperature_2m", [])
            precs = hh.get("precipitation", [])
            winds = hh.get("wind_speed_10m", [])
            idx = None
            for i, t in enumerate(times):
                if t.startswith(date_iso) and t.endswith(f"{hour_local:02d}:00"):
                    idx = i; break
            if idx is None:
                for i, t in enumerate(times):
                    if t.startswith(date_iso):
                        idx = i; break
            if idx is None:
                return {}
            temp = temps[idx] if idx < len(temps) else None
            precip = precs[idx] if idx < len(precs) else None
            wind = winds[idx] if idx < len(winds) else None
            return {
                "temp_c": temp,
                "precip_mm": precip,
                "wind_kmh": wind,
                "rain": (precip or 0) >= 1.0,
                "snow": False,
                "wind_strong": (wind or 0) >= 25,
                "hot": (temp or 0) >= 30,
                "cold": (temp or 99) <= 0,
            }
        except Exception:
            return {}

    # ---- fallback modeling ----
    def combine_strengths(team_for, league_for, opp_against, league_against, league_mean, home_adj=1.0):
        vals = [team_for, league_for, opp_against, league_against, league_mean]
        if any(v is None or (isinstance(v,float) and np.isnan(v)) for v in vals):
            return None
        f = team_for / max(1e-9, league_for)
        g = opp_against / max(1e-9, league_against)
        lam = f * g * league_mean * home_adj
        return float(max(0.05, lam))

    def blended_var_factor(v_for, v_against, league_vr, lo=1.0, hi=2.0):
        vals = [x for x in [v_for, v_against, league_vr] if isinstance(x,(int,float)) and x>0]
        if not vals: return 1.2
        v = float(np.median(vals))
        return float(min(max(v, lo), hi))

    def _poisson_cdf(k, lam):
        if lam <= 0:
            return 1.0 if k >= 0 else 0.0
        p = exp(-lam)
        s = p
        for x in range(0, int(k)):
            p *= lam / (x+1)
            s += p
            if s >= 1.0: return 1.0
        return min(1.0, s)

    def _neg_bin_cdf(k, lam, var_ratio):
        if lam <= 0: return 0.0
        if var_ratio <= 1.05:
            return float(_poisson_cdf(k, lam))
        var = lam * var_ratio
        p = lam / var
        if p <= 0 or p >= 1:
            return float(_poisson_cdf(k, lam))
        r = lam * p / (1 - p)
        def log_pmf(x):
            return (lgamma(x+r) - lgamma(r) - lgamma(x+1)) + r*log(p) + x*log(1-p)
        m = min(int(k)+60, int(max(5*lam, k+60)))
        mx = None; logs = []
        for x in range(0, m+1):
            lp = log_pmf(x); logs.append(lp); mx = lp if (mx is None or lp > mx) else mx
        s = sum(exp(l - mx) for l in logs[:int(k)+1])
        z = sum(exp(l - mx) for l in logs)
        return float(s / max(1e-12, z))

    def finalize_probability(k_int, lam, var_factor=1.2, prefer_nb=True):
        if lam is None or lam <= 0:
            return 0.0
        k = int(k_int) - 1
        cdf = _neg_bin_cdf(k, lam, var_factor) if prefer_nb else float(_poisson_cdf(k, lam))
        return float(max(0.0, min(1.0, 1.0 - cdf)))

# ---------- pagina ----------
st.set_page_config(page_title="v8.5 • Probabilità calibrate + Meteo", layout="wide")
st.title("v8.5 • Probabilità calibrate + Meteo • Dashboard automatica")

# ---------- Paths ----------
DATA_PATH = Path('app/public/data/league_team_metrics.json')   # (non obbligatori in v8.x)
CAL_PATH  = Path('app/public/data/calibrators.json')
FIX_CACHE = Path('app/public/data/fixtures_cache.json')
FIX_CACHE.parent.mkdir(parents=True, exist_ok=True)

# ---------- Secrets ----------
API_KEY_FD = None
API_FOOTBALL_KEY = None
try:
    if "FOOTBALL_DATA_API_KEY" in st.secrets:
        API_KEY_FD = str(st.secrets["FOOTBALL_DATA_API_KEY"]).strip()
    if "API_FOOTBALL_KEY" in st.secrets:
        API_FOOTBALL_KEY = str(st.secrets["API_FOOTBALL_KEY"]).strip()
except Exception:
    pass

# ---------- Config ----------
try:
    with open('settings.yaml','r',encoding='utf-8') as f:
        CFG = yaml.safe_load(f)
except Exception:
    CFG = {"staleness_hours": 18, "default_thresholds":{"shots":[8,10,12,14],"corners":[3,4,5,6]}}

# ---------- Utility ----------
def _safe_df(): return pd.DataFrame(columns=["date","home","away"])

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

# alias → target: nomi come in football-data.co.uk (metriche stagione scorsa)
ALIASES_METRICS = {
    # La Liga (estesi)
    ("SP1","Athletic Club"): "Ath Bilbao",
    ("SP1","Athletic Bilbao"): "Ath Bilbao",
    ("SP1","Club Atlético de Madrid"): "Atl Madrid",
    ("SP1","Atletico Madrid"): "Atl Madrid",
    ("SP1","Atlético Madrid"): "Atl Madrid",
    ("SP1","FC Barcelona"): "Barcelona",
    ("SP1","Futbol Club Barcelona"): "Barcelona",
    ("SP1","Barcelona"): "Barcelona",
    ("SP1","Real Madrid CF"): "Real Madrid",
    ("SP1","Real Madrid"): "Real Madrid",
    ("SP1","Real Sociedad de Fútbol"): "Sociedad",
    ("SP1","Real Sociedad de Futbol"): "Sociedad",
    ("SP1","Real Sociedad"): "Sociedad",
    ("SP1","RCD Espanyol de Barcelona"): "Espanyol",
    ("SP1","Espanyol"): "Espanyol",
    ("SP1","Elche CF"): "Elche",
    ("SP1","Elche"): "Elche",
    ("SP1","Levante UD"): "Levante",
    ("SP1","Levante"): "Levante",
    ("SP1","RCD Mallorca"): "Mallorca",
    ("SP1","Mallorca"): "Mallorca",
    ("SP1","RC Celta de Vigo"): "Celta",
    ("SP1","Celta de Vigo"): "Celta",
    ("SP1","Celta Vigo"): "Celta",
    ("SP1","RC Deportivo de La Coruña"): "La Coruna",
    ("SP1","Real Oviedo"): "Oviedo",
    ("SP1","Oviedo"): "Oviedo",
    ("SP1","Real Betis"): "Betis",
    ("SP1","Real Sociedad San Sebastian"): "Sociedad",

    # Serie A
    ("I1","Internazionale"): "Inter",
    ("I1","SS Lazio"): "Lazio",
    ("I1","AS Roma"): "Roma",
    ("I1","AC Milan"): "AC Milan",
    ("I1","Hellas Verona"): "Verona",
    ("I1","US Sassuolo"): "Sassuolo",
    ("I1","Udinese Calcio"): "Udinese",
    ("I1","US Salernitana"): "Salernitana",
    ("I1","SSC Napoli"): "Napoli",
    ("I1","ACF Fiorentina"): "Fiorentina",

    # Bundesliga (estesi, inclusi HSV/Köln/Mainz/Heidenheim)
    ("D1","FC Bayern München"): "Bayern Munich",
    ("D1","Bayern München"): "Bayern Munich",
    ("D1","Bayern Munich"): "Bayern Munich",
    ("D1","Borussia Dortmund"): "Dortmund",
    ("D1","Bayer 04 Leverkusen"): "Leverkusen",
    ("D1","Bayer Leverkusen"): "Leverkusen",
    ("D1","Borussia Mönchengladbach"): "M'gladbach",
    ("D1","Borussia Moenchengladbach"): "M'gladbach",
    ("D1","Borussia Monchengladbach"): "M'gladbach",
    ("D1","1. FC Köln"): "FC Koln",
    ("D1","1. FC Koeln"): "FC Koln",
    ("D1","1 FC Koln"): "FC Koln",
    ("D1","FC Koln"): "FC Koln",
    ("D1","VfB Stuttgart"): "Stuttgart",
    ("D1","VfL Wolfsburg"): "Wolfsburg",
    ("D1","Wolfsburg"): "Wolfsburg",
    ("D1","TSG Hoffenheim"): "Hoffenheim",
    ("D1","1899 Hoffenheim"): "Hoffenheim",
    ("D1","Eintracht Frankfurt"): "Frankfurt",
    ("D1","SC Freiburg"): "Freiburg",
    ("D1","FC Augsburg"): "Augsburg",
    ("D1","RB Leipzig"): "RB Leipzig",
    ("D1","1. FC Union Berlin"): "Union Berlin",
    ("D1","FSV Mainz 05"): "Mainz",
    ("D1","Mainz 05"): "Mainz",
    ("D1","1. FC Heidenheim"): "Heidenheim",
    ("D1","1. FC Heidenheim 1846"): "Heidenheim",
    ("D1","Heidenheim 1846"): "Heidenheim",
    ("D1","FC Heidenheim 1846"): "Heidenheim",
    ("D1","1 FC Heidenheim 1846"): "Heidenheim",
    ("D1","VfL Bochum"): "Bochum",
    ("D1","SV Werder Bremen"): "Werder Bremen",
    ("D1","Werder Bremen"): "Werder Bremen",
    ("D1","Hamburger SV"): "Hamburg",
    ("D1","Hamburg SV"): "Hamburg",
    ("D1","Hamburg"): "Hamburg",
}

def _canon(s):
    s = strip_accents(s).lower()
    # normalizzazioni utili
    s = s.replace("muenchen","munich").replace("munchen","munich")  # DE
    s = s.replace("monchengladbach","mgladbach").replace("mönchengladbach","mgladbach")
    s = s.replace("koln","koln").replace("cologne","koln")
    s = s.replace("hamburger","hamburg")  # HSV
    # ES: semplificazioni comuni
    s = s.replace("futbol", "futbol").replace("atletico", "atletico")
    # rimuovi numeri (Schalke 04, 1846, 05…)
    s = re.sub(r'\b\d+\b', ' ', s)
    # pulizia caratteri
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    # rimuovi prefissi/comuni (aggiunti rc/rcd/futbol)
    s = re.sub(r'\b(cf|fc|ud|cd|sd|sv|rc|rcd|club|de|la|real|futbol|borussia|eintracht|tsg|vfl|vfb|sc)\b', '', s).strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def match_team_name(code, name, teams_available):
    # 1) alias espliciti
    key = (code, str(name).strip())
    if key in ALIASES_METRICS:
        return ALIASES_METRICS[key]
    # 2) match esatto
    if name in teams_available:
        return name
    # 3) fuzzy semplice (token Jaccard)
    target_map = {t: _canon(t) for t in teams_available}
    name_c = _canon(name)
    best_t, best_score = None, 0.0
    for t, c in target_map.items():
        if not c: continue
        set_a, set_b = set(name_c.split()), set(c.split())
        if not set_a or not set_b: continue
        inter = len(set_a & set_b)
        uni   = len(set_a | set_b)
        score = inter / max(1, uni)
        if score > best_score:
            best_t, best_score = t, score
    if best_t and best_score >= 0.34:
        return best_t
    return name  # fallback

def normalize_for_metrics(code, home, away, METRICS):
    teams = list(METRICS.get(code, {}).get('teams', {}).keys())
    if not teams:
        return home, away
    h = match_team_name(code, home, teams)
    a = match_team_name(code, away, teams)
    return h, a

# ---------- Cache fixtures su disco ----------
def _safe_cache_read():
    try:
        if FIX_CACHE.exists():
            return json.loads(FIX_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _cache_write(code, df, source):
    try:
        cache = _safe_cache_read()
        rows = [{"date": str(r["date"]), "home": str(r["home"]), "away": str(r["away"]), "source": source}
                for _, r in df.iterrows()]
        cache[code] = {"updated_at": int(time.time()), "rows": rows}
        FIX_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _cache_load(code, days):
    cache = _safe_cache_read()
    if code not in cache: return _safe_df(), "cache: vuota"
    rows = cache[code].get("rows", [])
    if not rows: return _safe_df(), "cache: vuota"
    today = date.today()
    horizon = today + timedelta(days=max(1, int(days)))
    keep = [r for r in rows if today <= date.fromisoformat(r["date"]) <= horizon]
    if not keep: return _safe_df(), "cache: nessuna data nell’orizzonte"
    df = pd.DataFrame(keep)
    return df[["date","home","away"]].drop_duplicates().reset_index(drop=True), ""

# ---------- FPL (Premier fallback) ----------
@st.cache_data(ttl=600, show_spinner=False)
def _fpl_fixtures_premier(days):
    try:
        teams = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json().get('teams', [])
        id2name = {t['id']: t['name'] for t in teams}
        fx = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=30).json()
        if not isinstance(fx, list): return _safe_df(), "FPL: risposta inattesa"
        today = date.today()
        horizon = today + timedelta(days=max(1, int(days)))
        rows = []
        offset = (datetime.datetime.now() - datetime.datetime.utcnow())
        for f in fx:
            kt = f.get('kickoff_time')
            if not kt: continue
            try:
                dt_utc = datetime.datetime.fromisoformat(kt.replace('Z', '+00:00'))
                d = (dt_utc + offset).date()
            except Exception:
                continue
            if d < today or d > horizon: continue
            h = id2name.get(f.get('team_h')); a = id2name.get(f.get('team_a'))
            if h and a: rows.append({"date": str(d), "home": h.strip(), "away": a.strip()})
        df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else _safe_df()
        if not df.empty: _cache_write("E0", df, "FPL")
        return df, ""
    except Exception as e:
        return _safe_df(), f"FPL err: {type(e).__name__}"

# ---------- football-data.org (fallback API) ----------
COMP_MAP = {"E0":"PL","I1":"SA","SP1":"PD","D1":"BL1"}

def _fd_call(days, api_key, shrink=False):
    today = date.today()
    horizon = today + timedelta(days=(5 if shrink else max(1, int(days))))
    params = {
        "dateFrom": str(today),
        "dateTo": str(horizon),
        "competitions": ",".join(COMP_MAP.values()),
        "status": "SCHEDULED,POSTPONED"
    }
    headers = {"X-Auth-Token": api_key}
    return requests.get("https://api.football-data.org/v4/matches", params=params, headers=headers, timeout=30)

def _fd_fixtures_all(days, api_key):
    if not api_key:
        return [], pd.DataFrame([{"Lega":"Tutte","Fonte":"—","Partite":0,"Errore":"Manca API key football-data"}])
    r = _fd_call(days, api_key, shrink=False)
    if r.status_code == 429:
        time.sleep(1.2)
        r = _fd_call(days, api_key, shrink=True)
    if r.status_code != 200:
        rows, diags = [], []
        for code, comp in COMP_MAP.items():
            try:
                params = {"dateFrom": str(date.today()),
                          "dateTo": str(date.today()+timedelta(days=max(1, int(min(days, 7))))),
                          "competitions": comp, "status": "SCHEDULED,POSTPONED"}
                rr = requests.get("https://api.football-data.org/v4/matches",
                                  params=params, headers={"X-Auth-Token": api_key}, timeout=30)
                if rr.status_code == 200:
                    js = rr.json(); cnt=0
                    for m in js.get("matches", []):
                        d = datetime.datetime.fromisoformat(m["utcDate"].replace("Z","+00:00")).date()
                        h = m["homeTeam"]["name"]; a = m["awayTeam"]["name"]
                        rows.append({"code": code,
                                     "league": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                                     "date": str(d), "home": h.strip(), "away": a.strip(), "source": "football-data"})
                        cnt+=1
                    diags.append({"Lega": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                                  "Fonte":"football-data", "Partite":cnt, "Errore":"—" if cnt>0 else "nessuna"})
                else:
                    diags.append({"Lega": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                                  "Fonte":"—", "Partite":0, "Errore": f"football-data HTTP {rr.status_code}"})
            except Exception as e:
                diags.append({"Lega": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                              "Fonte":"—", "Partite":0, "Errore": type(e).__name__})
        rows = sorted(rows, key=lambda x: (x["date"], x["league"], x["home"]))
        return rows, pd.DataFrame(diags)

    js = r.json()
    rows = []
    diags = {"E0":0,"I1":0,"SP1":0,"D1":0}
    for m in js.get("matches", []):
        comp_code = None
        comp_id = m.get("competition", {}).get("code") or m.get("competition", {}).get("id")
        inv = {v:k for k,v in COMP_MAP.items()}
        if isinstance(comp_id, str) and comp_id in inv:
            comp_code = inv[comp_id]
        else:
            name = (m.get("competition", {}).get("name") or "").lower()
            if "premier" in name: comp_code = "E0"
            elif "serie a" in name: comp_code = "I1"
            elif "liga" in name: comp_code = "SP1"
            elif "bundes" in name: comp_code = "D1"
        if not comp_code: continue
        d = datetime.datetime.fromisoformat(m["utcDate"].replace("Z","+00:00")).date()
        h = m["homeTeam"]["name"]; a = m["awayTeam"]["name"]
        rows.append({"code": comp_code,
                     "league": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[comp_code],
                     "date": str(d), "home": h.strip(), "away": a.strip(), "source": "football-data"})
        diags[comp_code]+=1

    diag_df = pd.DataFrame([
        {"Lega":"Premier League","Fonte":"football-data","Partite":diags["E0"],"Errore":"—" if diags["E0"]>0 else "nessuna"},
        {"Lega":"Serie A","Fonte":"football-data","Partite":diags["I1"],"Errore":"—" if diags["I1"]>0 else "nessuna"},
        {"Lega":"La Liga","Fonte":"football-data","Partite":diags["SP1"],"Errore":"—" if diags["SP1"]>0 else "nessuna"},
        {"Lega":"Bundesliga","Fonte":"football-data","Partite":diags["D1"],"Errore":"—" if diags["D1"]>0 else "nessuna"},
    ])
    rows = sorted(rows, key=lambda x: (x["date"], x["league"], x["home"]))
    return rows, diag_df

# ---------- API-FOOTBALL (primaria) ----------
API_F_IDS = {"E0":39,"I1":135,"SP1":140,"D1":78}

@st.cache_data(ttl=600, show_spinner=False)
def _api_football_fixtures_all(days, api_key):
    if not api_key:
        return [], pd.DataFrame([{"Lega":"Tutte","Fonte":"—","Partite":0,"Errore":"Manca API_FOOTBALL_KEY"}])
    today = date.today()
    horizon = today + timedelta(days=max(1,int(days)))
    from_str, to_str = str(today), str(horizon)
    season = today.year if today.month>=7 else (today.year-1)
    rows=[]; diags=[]
    headers = {"x-apisports-key": api_key}
    for code, lid in API_F_IDS.items():
        try:
            params = {"league": lid, "season": season, "from": from_str, "to": to_str}
            r = requests.get("https://v3.football.api-sports.io/fixtures", params=params, headers=headers, timeout=30)
            if r.status_code != 200:
                diags.append({"Lega": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                              "Fonte":"—", "Partite":0, "Errore": f"API-FOOTBALL HTTP {r.status_code}"})
                continue
            js = r.json(); cnt=0
            for it in js.get("response", []):
                d = date.fromisoformat(it["fixture"]["date"][:10])
                if d < today or d > horizon: continue
                h = it["teams"]["home"]["name"]; a = it["teams"]["away"]["name"]
                rows.append({"code": code,
                             "league": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                             "date": str(d), "home": h.strip(), "away": a.strip(), "source": "API-FOOTBALL"})
                cnt+=1
            diags.append({"Lega": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                          "Fonte":"API-FOOTBALL", "Partite":cnt, "Errore":"—" if cnt>0 else "nessuna"})
        except Exception as e:
            diags.append({"Lega": {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}[code],
                          "Fonte":"—", "Partite":0, "Errore": type(e).__name__})
    rows = sorted(rows, key=lambda x: (x["date"], x["league"], x["home"]))
    return rows, pd.DataFrame(diags)

# ---------- Metriche (stagione scorsa + opzionale corrente 30%) ----------
@st.cache_data(ttl=6*3600)
def _fd_stats_csv(season_yyzz: str, code: str) -> pd.DataFrame:
    url = f"https://www.football-data.co.uk/mmz4281/{season_yyzz}/{code}.csv"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) < 2000:
            return pd.DataFrame()
        try:
            df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1")
        except Exception:
            df = pd.read_csv(io.BytesIO(r.content))
        return df
    except Exception:
        return pd.DataFrame()

def _season_yyzz_today():
    today = date.today()
    start_year = today.year if today.month >= 7 else (today.year - 1)
    end_year = start_year + 1
    return f"{str(start_year)[-2:]}{str(end_year)[-2:]}", start_year, end_year

def _metrics_build(only_last=True, include_cur_weight=0.3):
    yyzz, sy, ey = _season_yyzz_today()
    last = f"{str(sy-1)[-2:]}{str(sy)[-2:]}"
    cur  = yyzz

    def build(df_raw):
        need = ["HomeTeam","AwayTeam","HS","AS","HC","AC"]
        if df_raw.empty or not all(c in df_raw.columns for c in need):
            return pd.DataFrame(), {}, {}
        df = df_raw[need].copy()
        for c in ["HomeTeam","AwayTeam"]:
            df[c] = df[c].astype(str).str.strip()
        for c in ["HS","AS","HC","AC"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["HomeTeam","AwayTeam","HS","AS","HC","AC"])
        shots_home_for = df["HS"].mean(); shots_away_for = df["AS"].mean()
        corners_home_for = df["HC"].mean(); corners_away_for = df["AC"].mean()
        league_means = {
            "shots_home_for": float(shots_home_for),
            "shots_away_for": float(shots_away_for),
            "shots_home_against": float(shots_away_for),
            "shots_away_against": float(shots_home_for),
            "corners_home_for": float(corners_home_for),
            "corners_away_for": float(corners_away_for),
            "corners_home_against": float(corners_away_for),
            "corners_away_against": float(corners_home_for),
        }
        vr = {
            "shots_home_for": float((df["HS"].var(ddof=1) or shots_home_for)/max(1e-9, shots_home_for)),
            "shots_away_for": float((df["AS"].var(ddof=1) or shots_away_for)/max(1e-9, shots_away_for)),
            "corners_home_for": float((df["HC"].var(ddof=1) or corners_home_for)/max(1e-9, corners_home_for)),
            "corners_away_for": float((df["AC"].var(ddof=1) or corners_away_for)/max(1e-9, corners_away_for)),
        }
        rows = []
        teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
        for t in teams:
            rows.append({
                "team": t,
                "shots_for_home": float(df[df["HomeTeam"]==t]["HS"].mean()),
                "shots_for_away": float(df[df["AwayTeam"]==t]["AS"].mean()),
                "shots_against_home": float(df[df["HomeTeam"]==t]["AS"].mean()),
                "shots_against_away": float(df[df["AwayTeam"]==t]["HS"].mean()),
                "corners_for_home": float(df[df["HomeTeam"]==t]["HC"].mean()),
                "corners_for_away": float(df[df["AwayTeam"]==t]["AC"].mean()),
                "corners_against_home": float(df[df["HomeTeam"]==t]["AC"].mean()),
                "corners_against_away": float(df[df["AwayTeam"]==t]["HC"].mean()),
                "vr_shots_for_home": float((df[df["HomeTeam"]==t]["HS"].var(ddof=1) or 1.2)),
                "vr_shots_for_away": float((df[df["AwayTeam"]==t]["AS"].var(ddof=1) or 1.2)),
                "vr_shots_against_home": float((df[df["HomeTeam"]==t]["AS"].var(ddof=1) or 1.2)),
                "vr_shots_against_away": float((df[df["AwayTeam"]==t]["HS"].var(ddof=1) or 1.2)),
                "vr_corners_for_home": float((df[df["HomeTeam"]==t]["HC"].var(ddof=1) or 1.3)),
                "vr_corners_for_away": float((df[df["AwayTeam"]==t]["AC"].var(ddof=1) or 1.3)),
                "vr_corners_against_home": float((df[df["HomeTeam"]==t]["AC"].var(ddof=1) or 1.3)),
                "vr_corners_against_away": float((df[df["AwayTeam"]==t]["HC"].var(ddof=1) or 1.3)),
                "league_means": league_means
            })
        return pd.DataFrame(rows), league_means, vr

    LEAGUES = {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}
    full = {}
    for code in LEAGUES:
        last_df = _fd_stats_csv(last, code)
        cur_df  = _fd_stats_csv(cur, code)
        m_last, lm_last, vr_last = build(last_df)
        m_cur,  lm_cur,  vr_cur  = build(cur_df)

        if m_last.empty and m_cur.empty:
            full[code] = {"teams":{}, "league_means":{}, "league_var_ratio":{}}
            continue

        if only_last or m_cur.empty:
            base = m_last if not m_last.empty else m_cur
            full[code] = {
                "teams": {r["team"]: r for _, r in base.iterrows()},
                "league_means": lm_last if m_last is not None else lm_cur,
                "league_var_ratio": vr_last if m_last is not None else vr_cur
            }
        else:
            w_cur = include_cur_weight
            teams = sorted(set(m_last["team"]).union(set(m_cur["team"])))
            rows = []
            def W(xlast, xcur):
                if (xlast is None or pd.isna(xlast)) and (xcur is None or pd.isna(xcur)): return np.nan
                if xlast is None or pd.isna(xlast): return float(xcur)
                if xcur is None or pd.isna(xcur):  return float(xlast)
                return float((1-w_cur)*xlast + w_cur*xcur)
            for t in teams:
                a = m_last[m_last["team"]==t]; b = m_cur[m_cur["team"]==t]
                a = a.iloc[0] if len(a) else None
                b = b.iloc[0] if len(b) else None
                row = {"team": t}
                for k in ["shots_for_home","shots_for_away","shots_against_home","shots_against_away",
                          "corners_for_home","corners_for_away","corners_against_home","corners_against_away",
                          "vr_shots_for_home","vr_shots_for_away","vr_shots_against_home","vr_shots_against_away",
                          "vr_corners_for_home","vr_corners_for_away","vr_corners_against_home","vr_corners_against_away"]:
                    v_last = a[k] if a is not None and k in a else np.nan
                    v_cur  = b[k] if b is not None and k in b else np.nan
                    row[k] = W(v_last, v_cur)
                lm = {kk: W(lm_last.get(kk,np.nan), lm_cur.get(kk,np.nan)) for kk in set(list(lm_last.keys())+list(lm_cur.keys()))}
                row["league_means"] = lm
                rows.append(row)
            full[code] = {
                "teams": {r["team"]: r for r in rows},
                "league_means": rows[0]["league_means"],
                "league_var_ratio": vr_last
            }
    return full

# ---------- Fallback profilo squadra ----------
def _team_profile_or_fallback(METRICS, code, team):
    lg = METRICS.get(code, {})
    teams = lg.get('teams', {})
    if team in teams:
        return teams[team], False
    lm = lg.get('league_means', {})
    vr = lg.get('league_var_ratio', {})
    return {
        "shots_for_home": lm.get('shots_home_for', np.nan),
        "shots_for_away": lm.get('shots_away_for', np.nan),
        "shots_against_home": lm.get('shots_home_against', np.nan),
        "shots_against_away": lm.get('shots_away_against', np.nan),
        "corners_for_home": lm.get('corners_home_for', np.nan),
        "corners_for_away": lm.get('corners_away_for', np.nan),
        "corners_against_home": lm.get('corners_home_against', np.nan),
        "corners_against_away": lm.get('corners_away_against', np.nan),
        "vr_shots_for_home": vr.get('shots_home_for', 1.2),
        "vr_shots_for_away": vr.get('shots_away_for', 1.2),
        "vr_shots_against_home": vr.get('shots_home_for', 1.2),
        "vr_shots_against_away": vr.get('shots_away_for', 1.2),
        "vr_corners_for_home": vr.get('corners_home_for', 1.3),
        "vr_corners_for_away": vr.get('corners_away_for', 1.3),
        "vr_corners_against_home": vr.get('corners_home_for', 1.3),
        "vr_corners_against_away": vr.get('corners_away_for', 1.3),
    }, True

# ---------- Modeling helpers ----------
def compute_lambda_and_var(METRICS, code, home, away, metric):
    if code not in METRICS: return None
    lg = METRICS[code]
    league_means = lg.get('league_means', {})
    league_vr = lg.get('league_var_ratio', {})

    th, _ = _team_profile_or_fallback(METRICS, code, home)
    ta, _ = _team_profile_or_fallback(METRICS, code, away)

    if metric == "tiri":
        team_for_home = th.get('shots_for_home'); league_for_home = league_means.get('shots_home_for')
        opp_against_away = ta.get('shots_against_away'); league_against_away = league_means.get('shots_away_against')
        league_mean = (league_means.get('shots_home_for',np.nan) + league_means.get('shots_away_for',np.nan))/2.0
        team_for_away = ta.get('shots_for_away'); opp_against_home = th.get('shots_against_home')
        league_against_home = league_means.get('shots_home_against')
        H_home, H_away = 1.05, 0.95
        vr_home = blended_var_factor(th.get('vr_shots_for_home'), ta.get('vr_shots_against_away'), league_vr.get('shots_home_for', 1.1), 1.0, 2.0)
        vr_away = blended_var_factor(ta.get('vr_shots_for_away'), th.get('vr_shots_against_home'), league_vr.get('shots_away_for', 1.1), 1.0, 2.0)
    else:
        team_for_home = th.get('corners_for_home'); league_for_home = league_means.get('corners_home_for')
        opp_against_away = ta.get('corners_against_away'); league_against_away = league_means.get('corners_away_against')
        league_mean = (league_means.get('corners_home_for',np.nan) + league_means.get('corners_away_for',np.nan))/2.0
        team_for_away = ta.get('corners_for_away'); opp_against_home = th.get('corners_against_home')
        league_against_home = league_means.get('corners_home_against')
        H_home, H_away = 1.03, 0.97
        vr_home = blended_var_factor(th.get('vr_corners_for_home'), ta.get('vr_corners_against_away'), league_vr.get('corners_home_for', 1.3), 1.1, 2.5)
        vr_away = blended_var_factor(ta.get('vr_corners_for_away'), th.get('vr_corners_against_home'), league_vr.get('corners_away_for', 1.3), 1.1, 2.5)

    lam_home = combine_strengths(team_for_home, league_for_home, opp_against_away, league_against_away, league_mean, H_home)
    lam_away = combine_strengths(team_for_away, league_for_home, opp_against_home, league_against_home, H_away)
    return lam_home, lam_away, vr_home, vr_away

def apply_isotonic(CAL, code, metric, side_key, k_int, p):
    try:
        if not CAL: return p
        if metric == "tiri": metric_key = "shots"
        elif metric == "angoli": metric_key = "corners"
        elif metric == "angoli_tot": metric_key = "corners_total"
        else: return p
        node = CAL.get(code, {}).get(metric_key, {}).get(side_key, None)
        if not node: return p
        if isinstance(node, dict) and ("x" in node and "y" in node):
            xs, ys = node["x"], node["y"]
        else:
            key = str(int(k_int))
            entry = node.get(key) if isinstance(node, dict) else None
            if not entry and isinstance(node, dict):
                nums = [int(k) for k in node.keys() if str(k).isdigit()]
                if nums:
                    entry = node.get(str(min(nums, key=lambda kk: abs(kk - int(k_int)))))
            if not isinstance(entry, dict): return p
            xs = entry.get("x") or entry.get("xs") or entry.get("X")
            ys = entry.get("y") or entry.get("ys") or entry.get("Y")
        if not (isinstance(xs,(list,tuple)) and isinstance(ys,(list,tuple)) and len(xs)==len(ys) and len(xs)>=2):
            return p
        pairs = sorted(zip(xs, ys), key=lambda t: t[0])
        xs = [float(a) for a,_ in pairs]; ys = [float(b) for _,b in pairs]
        if p <= xs[0]: return ys[0]
        if p >= xs[-1]: return ys[-1]
        import bisect
        j = bisect.bisect_right(xs, p) - 1
        j = max(0, min(j, len(xs)-2))
        x0,x1 = xs[j], xs[j+1]; y0,y1 = ys[j], ys[j+1]
        t = 0 if x1==x0 else (p-x0)/(x1-x0)
        return y0 + t*(y1-y0)
    except Exception:
        return p

def adjust_for_weather(lam, metric_name, flags):
    rain, snow, wind, hot, cold = flags
    if lam is None: return None
    if rain: lam *= 0.97 if metric_name=='tiri' else 1.06
    if snow: lam *= 0.94 if metric_name=='tiri' else 1.10
    if wind: lam *= 0.96 if metric_name=='tiri' else 1.08
    if hot:  lam *= 0.98
    if cold: lam *= 0.99 if metric_name=='tiri' else 1.01
    return lam

# ---------- Sidebar ----------
st.sidebar.header("Opzioni")
days = st.sidebar.slider("Giorni futuri (consigliato 7–10)", 1, 45, 10, 1)
kick_hour = st.sidebar.slider("Ora indicativa (meteo)", 12, 21, 18)
use_meteo = st.sidebar.checkbox("Meteo automatico", value=True)
include_current_season = st.sidebar.checkbox("Includi stagione corrente nelle metriche (30%)", value=False)

# ---------- Costruisci metriche ----------
with st.spinner("Costruisco metriche (stagione scorsa)..."):
    METRICS = _metrics_build(only_last=not include_current_season, include_cur_weight=0.3)

# ---------- Fixtures: API-FOOTBALL primaria, FD fallback, FPL per PL ----------
with st.spinner("Recupero partite (API-FOOTBALL)..."):
    fixtures, diag1 = _api_football_fixtures_all(days, API_FOOTBALL_KEY)

diag_frames = [diag1 if isinstance(diag1, pd.DataFrame) else pd.DataFrame(diag1)]
if not fixtures:
    fx_fd, diag_fd = _fd_fixtures_all(days, API_KEY_FD)
    fixtures = fx_fd
    diag_frames.append(diag_fd)

# Premier League: fallback FPL se necessario
if not any(f.get("code")=="E0" for f in fixtures):
    fpl_df, fpl_err = _fpl_fixtures_premier(days)
    if not fpl_df.empty:
        for _, r in fpl_df.iterrows():
            fixtures.append({"code":"E0","league":"Premier League","date":str(r["date"]),
                             "home":r["home"],"away":r["away"],"source":"FPL"})
        diag_frames.append(pd.DataFrame([{"Lega":"Premier League","Fonte":"FPL","Partite":len(fpl_df),"Errore":"—"}]))

diag = pd.concat(diag_frames, ignore_index=True) if len(diag_frames)>1 else diag_frames[0]

# Diagnostica
with st.expander("🔍 Diagnostica fonti partite"):
    st.dataframe(diag if isinstance(diag, pd.DataFrame) else pd.DataFrame(diag), use_container_width=True)

if not fixtures:
    st.warning("Nessuna partita trovata. Verifica le chiavi nei Secrets e riduci i giorni a 7–10.")
    st.stop()

# ---------- UI principale ----------
labels = [f"{f['league']} • {f['date']} • {f['home']} - {f['away']}  ({f['source']})" for f in fixtures]
pick = st.selectbox("Scegli una partita", labels, index=0)
fx = fixtures[labels.index(pick)]
code, home_raw, away_raw, date_iso = fx["code"], fx["home"], fx["away"], fx["date"]
tz = LEAGUE_TZ.get(code, "Europe/Rome")

# Mappa i nomi a quelli delle metriche
home, away = normalize_for_metrics(code, home_raw, away_raw, METRICS)

st.write(f"### {home_raw} vs {away_raw} — {fx['league']} — {date_iso}")
if (home != home_raw) or (away != away_raw):
    st.caption(f"Allineamento per metriche: {home_raw} → {home} • {away_raw} → {away}")

# Avviso se profilo sintetico
missing = []
if code in METRICS and METRICS[code].get('teams'):
    if home not in METRICS[code]['teams']: missing.append(home)
    if away not in METRICS[code]['teams']: missing.append(away)
if missing:
    st.info("Squadra senza storico nella lega: " + ", ".join(missing) +
            " → uso medie di campionato (fallback) finché non ci sono dati sufficienti.")

# Meteo
rain=snow=wind=hot=cold=False
if use_meteo:
    latlon = geocode_team_fallback(home_raw, code, autosave=True)
    if latlon and latlon.get("lat") and latlon.get("lon"):
        wx = fetch_openmeteo_conditions(latlon["lat"], latlon["lon"], date_iso, hour_local=kick_hour, tz=tz) or {}
        rain = wx.get('rain', False); snow = wx.get('snow', False)
        wind = wx.get('wind_strong', False); hot = wx.get('hot', False); cold = wx.get('cold', False)
        st.caption(f"Meteo: T={wx.get('temp_c','?')}°C • P={wx.get('precip_mm','?')}mm • V={wx.get('wind_kmh','?')}km/h")

c1, c2 = st.columns(2)
with c1:
    th_home = st.number_input(f"Soglia tiri {home} (usa .5, es. 10.5)", min_value=0.0, value=10.5, step=0.5)
with c2:
    th_away = st.number_input(f"Soglia tiri {away} (usa .5, es. 12.5)", min_value=0.0, value=12.5, step=0.5)

def k_from_half(th): return (int(th) + 1) if ((th % 1) == 0.5) else int(np.floor(th)) + 1

# Calcolo
shots_params   = compute_lambda_and_var(METRICS, code, home, away, "tiri")
corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")

if not shots_params:
    st.error("Dati squadra non disponibili per questa partita.")
    st.stop()

lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
lam_h_c=lam_a_c=vr_h_c=vr_a_c=None
if corners_params:
    lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params

# Meteo
if use_meteo:
    lam_h_s = adjust_for_weather(lam_h_s, 'tiri', (rain, snow, wind, hot, cold))
    lam_a_s = adjust_for_weather(lam_a_s, 'tiri', (rain, snow, wind, hot, cold))
    if lam_h_c is not None: lam_h_c = adjust_for_weather(lam_h_c, 'angoli', (rain, snow, wind, hot, cold))
    if lam_a_c is not None: lam_a_c = adjust_for_weather(lam_a_c, 'angoli', (rain, snow, wind, hot, cold))

kH = k_from_half(th_home)
kA = k_from_half(th_away)

pH = finalize_probability(kH, lam_h_s, var_factor=vr_h_s, prefer_nb=True)
pA = finalize_probability(kA, lam_a_s, var_factor=vr_a_s, prefer_nb=True)

# Calibratori (facoltativi)
CAL = {}
try:
    if CAL_PATH.exists():
        CAL = json.loads(CAL_PATH.read_text(encoding='utf-8'))
except Exception:
    CAL = {}

pH = apply_isotonic(CAL, code, "tiri", "home", kH, pH)
pA = apply_isotonic(CAL, code, "tiri", "away", kA, pA)

st.markdown(f"**Probabilità tiri – {home}**: Over {th_home} ➜ **{round(pH*100,1)}%**, Under {th_home} ➜ **{round((1-pH)*100,1)}%**")
st.markdown(f"**Probabilità tiri – {away}**: Over {th_away} ➜ **{round(pA*100,1)}%**, Under {th_away} ➜ **{round((1-pA)*100,1)}%**")

# Angoli totali
st.subheader("Angoli totali – Over standard")
if corners_params and lam_h_c and lam_a_c:
    lam_tot = float(lam_h_c + lam_a_c)
    var_tot = float(lam_h_c*vr_h_c + lam_a_c*vr_a_c)
    vf_tot  = (var_tot/lam_tot) if lam_tot > 0 else 1.0
    rows = []
    for th in [5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5]:
        k = int(np.floor(th)) + 1
        p = finalize_probability(k, lam_tot, var_factor=vf_tot, prefer_nb=True)
        p = apply_isotonic(CAL, code, "angoli_tot", "total", k, p)
        rows.append({"Soglia": f"Over {th}", "Probabilità": f"{round(p*100,1)}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("Dati corner incompleti: mostro solo tiri.")

# Lista completa prossimi giorni (veloce)
st.subheader("📊 Prossimi giorni – soglie standard (veloce)")
rows = []
for fx in fixtures:
    code0, home0, away0, d = fx["code"], fx["home"], fx["away"], fx["date"]
    h_m, a_m = normalize_for_metrics(code0, home0, away0, METRICS)
    sp = compute_lambda_and_var(METRICS, code0, h_m, a_m, "tiri")
    cp = compute_lambda_and_var(METRICS, code0, h_m, a_m, "angoli")
    if not sp: continue
    lam_h_s, lam_a_s, vr_h_s, vr_a_s = sp
    row = {"Lega": fx["league"], "Data": d, "Match": f"{home0}-{away0}"}
    for k in [8,10,12,14]:
        ph = finalize_probability(k, lam_h_s, var_factor=vr_h_s, prefer_nb=True)
        pa = finalize_probability(k, lam_a_s, var_factor=vr_a_s, prefer_nb=True)
        ph = apply_isotonic(CAL, code0, "tiri", "home", k, ph)
        pa = apply_isotonic(CAL, code0, "tiri", "away", k, pa)
        row[f"tiri_H≥{k}"] = round(ph*100,1)
        row[f"tiri_A≥{k}"] = round(pa*100,1)
    if cp:
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = cp
        for k in [3,4,5,6]:
            phc = finalize_probability(k, lam_h_c, var_factor=vr_h_c, prefer_nb=True)
            pac = finalize_probability(k, lam_a_c, var_factor=vr_a_c, prefer_nb=True)
            phc = apply_isotonic(CAL, code0, "angoli", "home", k, phc)
            pac = apply_isotonic(CAL, code0, "angoli", "away", k, pac)
            row[f"angoli_H≥{k}"] = round(phc*100,1)
            row[f"angoli_A≥{k}"] = round(pac*100,1)
    rows.append(row)

if rows:
    df_out = pd.DataFrame(rows)
    try:
        df_out["Data_s"] = pd.to_datetime(df_out["Data"], errors="coerce")
        df_out = df_out.sort_values(["Data_s","Lega","Match"]).drop(columns=["Data_s"])
    except Exception:
        pass
    st.dataframe(df_out, use_container_width=True)
    st.download_button("Scarica CSV unico", df_out.to_csv(index=False).encode('utf-8'),
                       "dashboard_prob_calibrate.csv", "text/csv")
else:
    st.info("Nessuna riga da mostrare nella tabella veloce.")

st.caption("Calendari: API-FOOTBALL (primaria) • Fallback: football-data.org/FPL • Meteo: Open-Meteo • Metriche: football-data.co.uk (stagione scorsa; fallback medie per neopromosse).")
