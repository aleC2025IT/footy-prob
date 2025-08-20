# -*- coding: utf-8 -*-
# ===== streamlit_app.py — v7 FIX TOTALE (file unico) =====
# Mostra tutte le leghe (Serie A / La Liga / Bundesliga / Premier) con WorldFootball.
# Ha fallback completi se i moduli 'pipeline' non esistono:
#  - Modello probabilistico Poisson/NegBin semplificato interno
#  - Costruzione METRICS da football-data.co.uk (stagione scorsa + corrente)
#  - Meteo via Open-Meteo senza chiavi
#
# Se i tuoi JSON/pipepline ci sono, li usa. Se non ci sono, funziona lo stesso.

# --- make repo root importable ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# stdlib
import json, re, time, datetime
from datetime import timedelta, date
from pathlib import Path
import unicodedata

# third-party
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yaml
from scipy.stats import poisson  # per il fallback
import io

# ====== TENTATIVO IMPORT PIPELINE (con fallback) ======
HAVE_PIPELINE = True
try:
    from pipeline.modeling.prob_model import combine_strengths, finalize_probability, blended_var_factor
    from pipeline.utils.auto_weather import fetch_openmeteo_conditions, LEAGUE_TZ
    from pipeline.utils.geocode import geocode_team_fallback
except Exception:
    HAVE_PIPELINE = False

    # ----- Fallback meteo / geocode -----
    LEAGUE_TZ = {"I1":"Europe/Rome","E0":"Europe/London","SP1":"Europe/Madrid","D1":"Europe/Berlin"}

    def geocode_team_fallback(name, code, autosave=True):
        # usa Open-Meteo geocoding libero
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
        # semplificazione: prendiamo l’ora 17:00 locale se c’è, altrimenti la prima della data
        try:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "forecast_days": 7,
                "timezone": "auto"
            }
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
            js = r.json()
            hourly = js.get("hourly", {})
            times = hourly.get("time", [])
            temps = hourly.get("temperature_2m", [])
            precs = hourly.get("precipitation", [])
            winds = hourly.get("wind_speed_10m", [])
            target = date_iso
            idx = None
            # cerchiamo ore della stessa data
            for i, t in enumerate(times):
                if t.startswith(target) and t.endswith(f"{hour_local:02d}:00"):
                    idx = i; break
            if idx is None:
                for i, t in enumerate(times):
                    if t.startswith(target):
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

    # ----- Fallback modeling -----
    def combine_strengths(team_for, league_for, opp_against, league_against, league_mean, home_adj=1.0):
        # prodotto normalizzato semplice + aggiustamento casa/trasferta
        if any(pd.isna([team_for, league_for, opp_against, league_against, league_mean])):
            return None
        f = team_for / max(1e-9, league_for)
        g = opp_against / max(1e-9, league_against)
        lam = f * g * league_mean * home_adj
        return float(max(0.05, lam))

    def blended_var_factor(v_for, v_against, league_vr, lo=1.0, hi=2.0):
        # se non ci sono varianze, usa ratio di lega o 1.2
        vals = [x for x in [v_for, v_against, league_vr] if isinstance(x,(int,float)) and x>0]
        if not vals: return 1.2
        v = float(np.median(vals))
        return float(min(max(v, lo), hi))

    def _neg_bin_cdf(k, lam, var_ratio):
        # usa Poisson se var_ratio ~1, altrimenti NegBin approssimata
        if lam <= 0: return 0.0
        if var_ratio <= 1.05:
            return float(poisson.cdf(k, lam))
        # NegBin parametrizzazione: mean=lam, var = lam * var_ratio => p = mean/var, r = mean*p/(1-p)
        var = lam * var_ratio
        p = lam / var
        if p <= 0 or p >= 1:  # fallback
            return float(poisson.cdf(k, lam))
        r = lam * p / (1 - p)
        # CDF NB(k; r, p) = I_{p}(r, k+1) (incomplete beta). Evitiamo dipendenze: approssimiamo sommando PMF
        # PMF: C(k+r-1, k) * (1-p)^k * p^r
        from math import lgamma, exp, log
        def log_pmf(x):
            return (lgamma(x+r) - lgamma(r) - lgamma(x+1)) + r*log(p) + x*log(1-p)
        # somma fino a k (può essere costosa per k grande; per soglie qui va bene)
        m = min(int(k)+60, int(max(5*lam, k+60)))  # taglia code
        # normalizzazione
        # uso accumulo in spazio log per stabilità
        mx = None
        logs = []
        for x in range(0, m+1):
            lp = log_pmf(x)
            logs.append(lp)
            mx = lp if (mx is None or lp > mx) else mx
        s = sum(exp(l - mx) for l in logs[:int(k)+1])
        z = sum(exp(l - mx) for l in logs)
        return float(s / max(1e-12, z))

    def finalize_probability(k_int, lam, var_factor=1.2, prefer_nb=True):
        # ritorna P(X >= k_int)
        if lam is None or lam <= 0:
            return 0.0
        # CDF fino a k-1
        k = int(k_int) - 1
        if prefer_nb:
            cdf = _neg_bin_cdf(k, lam, var_factor)
        else:
            cdf = float(poisson.cdf(k, lam))
        return float(max(0.0, min(1.0, 1.0 - cdf)))

# ---------- Page ----------
st.set_page_config(page_title="v7 • Probabilità calibrate + Meteo", layout="wide")
st.title("v7 • Probabilità calibrate + Meteo • Dashboard automatica")

# ---------- Paths ----------
DATA_PATH = Path('app/public/data/league_team_metrics.json')
CAL_PATH  = Path('app/public/data/calibrators.json')
FIX_CACHE = Path('app/public/data/fixtures_cache.json')
FIX_CACHE.parent.mkdir(parents=True, exist_ok=True)

# ---------- Secrets / API key (opzionale) ----------
API_KEY = None
try:
    if "FOOTBALL_DATA_API_KEY" in st.secrets:
        API_KEY = st.secrets["FOOTBALL_DATA_API_KEY"].strip()
except Exception:
    pass

# ---------- Helpers base ----------
@st.cache_data(show_spinner=False)
def load_json_cached(path):
    p = Path(path)
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def data_age_hours(path: Path) -> float:
    p = Path(path)
    if not p.exists():
        return 1e9
    return (datetime.datetime.now().timestamp() - p.stat().st_mtime) / 3600.0

def run_full_update():
    # Se pipeline esiste, lanciamo i moduli; altrimenti segnaliamo ok (non bloccare app)
    if HAVE_PIPELINE:
        import subprocess
        r1 = subprocess.run([sys.executable, "-m", "pipeline.export_json"], cwd=ROOT, capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-m", "pipeline.build_calibrators"], cwd=ROOT, capture_output=True, text=True)
        ok = True
        if r1.returncode != 0:
            st.error("Errore metriche: " + (r1.stderr or r1.stdout)[-500:])
            ok = False
        if r2.returncode != 0:
            st.error("Errore calibratori: " + (r2.stderr or r2.stdout)[-500:])
            ok = False
        st.cache_data.clear()
        return ok
    else:
        st.info("Pipeline non presente: uso metriche fallback automatiche da football-data.co.uk.")
        st.cache_data.clear()
        return True

def auto_update_if_stale(max_age_hours=18):
    stale_metrics = data_age_hours(DATA_PATH) > max_age_hours
    stale_cal     = data_age_hours(CAL_PATH)  > max_age_hours
    if stale_metrics or stale_cal:
        with st.spinner("Aggiornamento automatico dei dati in corso..."):
            ok = run_full_update()
            if ok:
                st.success("Aggiornamento completato. Se la tabella non appare, premi Rerun.")

# ---------- Fixtures – mappe / helper ----------
LEAGUE_NAMES = {"I1":"Serie A","E0":"Premier League","SP1":"La Liga","D1":"Bundesliga"}
ALL_CODES = ["I1","E0","SP1","D1"]

def _safe_df():
    return pd.DataFrame(columns=["date","home","away"])

# ---------- Cache su disco (persistente) ----------
def _read_fix_cache():
    try:
        if FIX_CACHE.exists():
            return json.loads(FIX_CACHE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _write_fix_cache(code, df, source):
    try:
        cache = _read_fix_cache()
        rows = [{"date": str(r["date"]), "home": str(r["home"]), "away": str(r["away"]), "source": source}
                for _, r in df.iterrows()]
        cache[code] = {"updated_at": int(time.time()), "rows": rows}
        FIX_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def _load_from_fix_cache(code, days):
    cache = _read_fix_cache()
    if code not in cache:
        return _safe_df(), "cache: vuota"
    rows = cache[code].get("rows", [])
    if not rows:
        return _safe_df(), "cache: vuota"
    today = date.today()
    horizon = today + timedelta(days=max(1, int(days)))
    keep = [r for r in rows if today <= date.fromisoformat(r["date"]) <= horizon]
    if not keep:
        return _safe_df(), "cache: nessuna data nell’orizzonte"
    df = pd.DataFrame(keep)
    return df[["date","home","away"]].drop_duplicates().reset_index(drop=True), ""

# ---------- Normalizzazione nomi (per allineare WorldFootball ↔ metriche) ----------
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

ALIASES = {
    # La Liga
    ("SP1","Athletic Bilbao"): "Athletic Club",
    ("SP1","Atletico Madrid"): "Atlético Madrid",
    ("SP1","Real Betis Sevilla"): "Real Betis",
    ("SP1","Deportivo Alaves"): "Alavés",
    ("SP1","Cadiz CF"): "Cádiz",
    ("SP1","Real Sociedad San Sebastian"): "Real Sociedad",
    ("SP1","Celta de Vigo"): "Celta Vigo",
    ("SP1","Girona FC"): "Girona",
    ("SP1","Granada CF"): "Granada",
    ("SP1","Sevilla FC"): "Sevilla",
    # Serie A
    ("I1","Internazionale"): "Inter",
    ("I1","AS Roma"): "Roma",
    ("I1","SS Lazio"): "Lazio",
    ("I1","US Lecce"): "Lecce",
    ("I1","US Sassuolo"): "Sassuolo",
    ("I1","Hellas Verona"): "Hellas Verona",
    ("I1","Udinese Calcio"): "Udinese",
    ("I1","ACF Fiorentina"): "Fiorentina",
    ("I1","US Salernitana"): "Salernitana",
    ("I1","SSC Napoli"): "Napoli",
    # Bundesliga
    ("D1","Borussia Moenchengladbach"): "Borussia Mönchengladbach",
    ("D1","1. FC Koeln"): "Köln",
    ("D1","1. FC Union Berlin"): "Union Berlin",
    ("D1","FSV Mainz 05"): "Mainz 05",
}

def normalize_name(code, name):
    name_clean = re.sub(r'\s+', ' ', str(name)).strip()
    key = (code, name_clean)
    if key in ALIASES:
        return ALIASES[key]
    s = strip_accents(name_clean)
    s = re.sub(r'\b(CF|FC|UD|CD|SD)\b\.?', '', s, flags=re.I).strip()
    s = re.sub(r'\s+', ' ', s)
    if code=="SP1" and s.lower()=="athletic bilbao": return "Athletic Club"
    if code=="D1" and "Monchengladbach" in s: return "Borussia Mönchengladbach"
    if code=="I1" and s.lower()=="internazionale": return "Inter"
    return name_clean

# ---------- FONTE 0: FPL (Premier) ----------
@st.cache_data(ttl=600, show_spinner=False)
def _fpl_fixtures_premier(days):
    try:
        teams = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json().get('teams', [])
        id2name = {t['id']: t['name'] for t in teams}
        fx = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=30).json()
        if not isinstance(fx, list):
            return _safe_df(), "FPL: risposta inattesa"
        today = date.today()
        horizon = today + timedelta(days=max(1, int(days)))
        rows = []
        # conversione UTC->locale semplice
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
        if not df.empty: _write_fix_cache("E0", df, "FPL")
        return df, ""
    except Exception as e:
        return _safe_df(), f"FPL err: {type(e).__name__}"

# ---------- FONTE 1: WorldFootball (Serie A, La Liga, Bundesliga) ----------
WF_SLUG = {"I1":"ita-serie-a", "SP1":"esp-primera-division", "D1":"bundesliga"}

def _season_slug_today():
    today = date.today()
    start_year = today.year if today.month >= 7 else (today.year - 1)
    return f"{start_year}-{start_year+1}"

def _ua_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en,en-GB;q=0.9"
    })
    return s

@st.cache_data(ttl=900, show_spinner=False)
def _worldfootball_fixtures(code, days):
    """
    Legge i calendari da worldfootball provando PRIMA 'schedule' e poi 'all_matches'.
    Riconosce varie intestazioni (Home team/Home/Away team/Away), unisce tutte le tabelle e filtra per orizzonte futuro.
    """
    slug = WF_SLUG.get(code)
    if not slug:
        return _safe_df(), "WF: slug mancante"

    try:
        season = _season_slug_today()
        sess = _ua_session()
        urls = [
            f"https://www.worldfootball.net/schedule/{slug}-{season}/",
            f"https://www.worldfootball.net/all_matches/{slug}-{season}/",
        ]
        frames = []
        errs = []

        for url in urls:
            try:
                resp = sess.get(url, timeout=30)
                if resp.status_code != 200:
                    errs.append(f"{url.split('/')[3]} HTTP {resp.status_code}")
                    continue
                tables = pd.read_html(resp.text)
                for t in tables:
                    cols_lower = [str(c).strip().lower() for c in t.columns]
                    rename_map = {}
                    if "home team" in cols_lower: rename_map[t.columns[cols_lower.index("home team")]] = "home"
                    if "away team" in cols_lower: rename_map[t.columns[cols_lower.index("away team")]] = "away"
                    if "home" in cols_lower:      rename_map[t.columns[cols_lower.index("home")]]      = "home"
                    if "away" in cols_lower:      rename_map[t.columns[cols_lower.index("away")]]      = "away"
                    if "date" in cols_lower:      rename_map[t.columns[cols_lower.index("date")]]      = "date"
                    t = t.rename(columns=rename_map)
                    if not all(k in t.columns for k in ["date", "home", "away"]):
                        continue
                    tt = t[["date", "home", "away"]].copy()
                    tt["date"] = pd.to_datetime(tt["date"], dayfirst=True, errors="coerce").dt.date
                    tt = tt.dropna(subset=["date","home","away"])
                    if not tt.empty:
                        frames.append(tt)
            except Exception as e:
                errs.append(f"{url.split('/')[3]} {type(e).__name__}")

        if not frames:
            msg = "WF: nessuna tabella"
            if errs:
                msg += " | " + " | ".join(errs)
            return _safe_df(), msg

        df = pd.concat(frames, ignore_index=True).drop_duplicates()
        today = date.today()
        horizon = today + timedelta(days=max(1, int(days)))
        df = df[(df["date"] >= today) & (df["date"] <= horizon)]

        df["home"] = df["home"].astype(str).str.strip().map(lambda s: normalize_name(code, s))
        df["away"] = df["away"].astype(str).str.strip().map(lambda s: normalize_name(code, s))
        df = df.drop_duplicates().reset_index(drop=True)

        if df.empty:
            return _safe_df(), "WF: nessuna partita nell'orizzonte"

        _write_fix_cache(code, df, "WorldFootball")
        return df, ""

    except requests.HTTPError as e:
        return _safe_df(), f"WF HTTP {getattr(e.response, 'status_code', '')}"
    except Exception as e:
        return _safe_df(), f"WF err: {type(e).__name__}"

# ---------- FONTE 2: football-data.org (facoltativa) ----------
@st.cache_data(ttl=600, show_spinner=False)
def _football_data_fixtures(code, days, api_key):
    comp = {"E0":"PL","SP1":"PD","D1":"BL1","I1":"SA"}.get(code.upper())
    if not comp or not api_key:
        return _safe_df(), "no key/comp"
    try:
        today = date.today()
        horizon = today + timedelta(days=max(1, int(days)))
        params = {"dateFrom": str(today), "dateTo": str(horizon), "competitions": comp, "status": "SCHEDULED,POSTPONED"}
        headers = {"X-Auth-Token": api_key}
        r = requests.get("https://api.football-data.org/v4/matches", params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            return _safe_df(), f"football-data HTTP {r.status_code}"
        js = r.json()
        rows = []
        for m in js.get("matches", []):
            utc_str = m.get("utcDate")
            if not utc_str: continue
            try:
                d = datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00")).date()
            except Exception:
                continue
            if d < today or d > horizon: continue
            h = m.get("homeTeam", {}).get("name"); a = m.get("awayTeam", {}).get("name")
            if h and a: rows.append({"date": str(d), "home": h.strip(), "away": a.strip()})
        df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else _safe_df()
        if not df.empty: _write_fix_cache(code, df, "football-data")
        return df, ""
    except Exception as e:
        return _safe_df(), f"football-data err: {type(e).__name__}"

# -------- Sticky cache (se oggi fallisce, tieni l'ultima lista buona) --------
def _sticky_merge(current_fixtures, current_diag):
    prev_fx  = st.session_state.get("last_fixtures", [])
    prev_diag = st.session_state.get("last_diag", None)
    for code, name in LEAGUE_NAMES.items():
        try:
            row = current_diag[current_diag["Lega"]==name].iloc[0]
            partite = int(row["Partite"]); errore = str(row["Errore"])
        except Exception:
            partite, errore = 0, ""
        if partite == 0 and errore != "—":
            prev_for_code = [f for f in prev_fx if f.get("code")==code]
            if prev_for_code:
                current_fixtures = [f for f in current_fixtures if f.get("code")!=code] + prev_for_code
                try:
                    idx = current_diag.index[current_diag["Lega"]==name][0]
                    current_diag.at[idx, "Fonte"] = "cache (session)"
                    current_diag.at[idx, "Partite"] = len(prev_for_code)
                    current_diag.at[idx, "Errore"] = errore + " → cache sessione"
                except Exception:
                    pass
    current_fixtures = sorted(current_fixtures, key=lambda x: (x["date"], x["league"], x["home"]))
    st.session_state["last_fixtures"] = current_fixtures
    st.session_state["last_diag"] = current_diag
    return current_fixtures, current_diag

# ---------- Funzione principale fixtures ----------
def get_all_fixtures(days=30, use_fd=False, use_local_cache=True, use_worldfootball=True):
    out, diags = [], []
    for code in ALL_CODES:
        df = _safe_df(); src=None; err=""

        # 0) Premier via FPL
        if code=="E0":
            d1, err1 = _fpl_fixtures_premier(days)
            if not d1.empty: df, src = d1, "FPL"
            else: err = err1

        # 1) WorldFootball (per I1/SP1/D1) – PRIMA SCELTA
        if df.empty and use_worldfootball and code in WF_SLUG:
            dWF, errWF = _worldfootball_fixtures(code, days)
            if not dWF.empty: df, src, err = dWF, "WorldFootball", ""
            else: err = (err + " | " + errWF).strip(" |")

        # 2) football-data (facoltativo)
        if df.empty and use_fd and API_KEY:
            d2, err2 = _football_data_fixtures(code, days, API_KEY)
            if not d2.empty: df, src, err = d2, "football-data", ""
            else: err = (err + " | " + err2).strip(" |")

        # 3) Cache su disco (persistente)
        if df.empty and use_local_cache:
            d4, err4 = _load_from_fix_cache(code, days)
            if not d4.empty: df, src, err = d4, "cache", (err + " | cache").strip(" |")
            else: err = (err + " | " + err4).strip(" |")

        diags.append({"Lega": LEAGUE_NAMES[code], "Fonte": src or "—", "Partite": len(df), "Errore": err or "—"})
        if not df.empty:
            for _, r in df.iterrows():
                out.append({"code": code, "league": LEAGUE_NAMES[code], "date": str(r["date"]),
                            "home": str(r["home"]).strip(), "away": str(r["away"]).strip(), "source": src or "—"})
    out = sorted(out, key=lambda x: (x["date"], x["league"], x["home"]))
    return out, pd.DataFrame(diags)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_fixtures(days, use_fd, use_local_cache, use_worldfootball):
    return get_all_fixtures(days, use_fd, use_local_cache, use_worldfootball)

# ====== FALLBACK METRICS (se mancano i JSON della pipeline) ======
LEAGUE_CODES_FD = {"E0":"Premier League","I1":"Serie A","SP1":"La Liga","D1":"Bundesliga"}

@st.cache_data(ttl=6*3600)
def _fd_fetch_csv(season_yyzz: str, code: str) -> pd.DataFrame:
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

def _metrics_from_fd():
    """Crea METRICS stile pipeline partendo dai CSV football-data della stagione scorsa + corrente."""
    yyzz, sy, ey = _season_yyzz_today()
    last = f"{str(sy-1)[-2:]}{str(sy)[-2:]}"
    cur  = yyzz

    full = {}
    for code in ALL_CODES:
        df_last = _fd_fetch_csv(last, code)
        df_cur  = _fd_fetch_csv(cur, code)
        if df_last.empty and df_cur.empty:
            full[code] = {"teams":{}, "league_means":{}, "league_var_ratio":{}}
            continue

        def build(df_raw):
            need = ["HomeTeam","AwayTeam","HS","AS","HC","AC"]
            if not all(c in df_raw.columns for c in need):
                return pd.DataFrame(), {}
            df = df_raw[need].copy()
            for c in ["HomeTeam","AwayTeam"]:
                df[c] = df[c].astype(str).str.strip()
            for c in ["HS","AS","HC","AC"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["HomeTeam","AwayTeam","HS","AS","HC","AC"])
            # medie
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
            # var ratio di lega (grossolana)
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
            return pd.DataFrame(rows), {"league_means": league_means, "league_var_ratio": vr}

        m_last, extra_last = build(df_last) if not df_last.empty else (pd.DataFrame(), {})
        m_cur,  extra_cur  = build(df_cur)  if not df_cur.empty  else (pd.DataFrame(), {})
        if m_last.empty and m_cur.empty:
            full[code] = {"teams":{}, "league_means":{}, "league_var_ratio":{}}
            continue

        # Fusione: 70% last + 30% current (semplice)
        if not m_last.empty and not m_cur.empty:
            w_cur = 0.3
            teams = sorted(set(m_last["team"]).union(set(m_cur["team"])))
            rows = []
            for t in teams:
                a = m_last[m_last["team"]==t]
                b = m_cur[m_cur["team"]==t]
                a = a.iloc[0] if len(a) else None
                b = b.iloc[0] if len(b) else None
                def W(xlast, xcur):
                    if (xlast is None or pd.isna(xlast)) and (xcur is None or pd.isna(xcur)): return np.nan
                    if xlast is None or pd.isna(xlast): return float(xcur)
                    if xcur is None or pd.isna(xcur):  return float(xlast)
                    return float((1-w_cur)*xlast + w_cur*xcur)
                row = {"team": t}
                for k in ["shots_for_home","shots_for_away","shots_against_home","shots_against_away",
                          "corners_for_home","corners_for_away","corners_against_home","corners_against_away",
                          "vr_shots_for_home","vr_shots_for_away","vr_shots_against_home","vr_shots_against_away",
                          "vr_corners_for_home","vr_corners_for_away","vr_corners_against_home","vr_corners_against_away"]:
                    v_last = a[k] if a is not None and k in a else np.nan
                    v_cur  = b[k] if b is not None and k in b else np.nan
                    row[k] = W(v_last, v_cur)
                # league means/var
                lm_last = extra_last.get("league_means", {})
                lm_cur  = extra_cur.get("league_means", {})
                league_means = {}
                for kk in set(list(lm_last.keys())+list(lm_cur.keys())):
                    league_means[kk] = W(lm_last.get(kk,np.nan), lm_cur.get(kk,np.nan))
                row["league_means"] = league_means
                rows.append(row)
            full[code] = {
                "teams": {r["team"]: r for r in rows},
                "league_means": rows[0]["league_means"],
                "league_var_ratio": extra_last.get("league_var_ratio", {})
            }
        else:
            base = m_cur if not m_cur.empty else m_last
            full[code] = {
                "teams": {r["team"]: r for _, r in base.iterrows()},
                "league_means": base.iloc[0]["league_means"],
                "league_var_ratio": (extra_cur or extra_last).get("league_var_ratio", {})
            }
    return full

# ---------- Modeling helpers ----------
def compute_lambda_and_var(METRICS, code, home, away, metric):
    if code not in METRICS: return None
    teams_obj = METRICS[code].get('teams', {})
    if home not in teams_obj or away not in teams_obj: return None
    lg = METRICS[code]
    league_means = lg.get('league_means', {})
    league_vr = lg.get('league_var_ratio', {})
    th = teams_obj[home]; ta = teams_obj[away]
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
    lam_away = combine_strengths(team_for_away, league_for_home, opp_against_home, league_against_home, league_mean, H_away)
    return lam_home, lam_away, vr_home, vr_away

def apply_isotonic(CAL, code, metric, side_key, k_int, p):
    if not CAL: return p
    if metric == "tiri": metric_key = "shots"
    elif metric == "angoli": metric_key = "corners"
    elif metric == "angoli_tot": metric_key = "corners_total"
    else: return p
    cal = CAL.get(code, {}).get(metric_key, {}).get(side_key, {})
    if not cal: return p
    key = str(int(k_int))
    if key not in cal:
        keys = sorted([int(x) for x in cal.keys()]) if cal else []
        if not keys: return p
        key = str(min(keys, key=lambda x: abs(x - int(k_int))))
    xs, ys = cal['x'], cal['y']
    if p <= xs[0]: return ys[0]
    if p >= xs[-1]: return ys[-1]
    import bisect
    j = bisect.bisect_right(xs, p) - 1
    j = max(0, min(j, len(xs)-2))
    x0,x1=xs[j], xs[j+1]; y0,y1=ys[j], ys[j+1]
    t = 0 if x1==x0 else (p-x0)/(x1-x0)
    return y0 + t*(y1-y0)

def adjust_for_weather(lam, metric_name, flags):
    rain, snow, wind, hot, cold = flags
    if lam is None: return None
    if rain: lam *= 0.97 if metric_name=='tiri' else 1.06
    if snow: lam *= 0.94 if metric_name=='tiri' else 1.10
    if wind: lam *= 0.96 if metric_name=='tiri' else 1.08
    if hot:  lam *= 0.98
    if cold: lam *= 0.99 if metric_name=='tiri' else 1.01
    return lam

# ---------- Settings & auto-update ----------
try:
    with open('settings.yaml','r',encoding='utf-8') as f:
        CFG = yaml.safe_load(f)
except Exception:
    CFG = {"staleness_hours": 18, "default_thresholds":{"shots":[8,10,12,14],"corners":[3,4,5,6]}}

auto_update_if_stale(CFG.get('staleness_hours', 18))

# Carica METRICS/CAL se presenti, altrimenti costruisci fallback
METRICS = load_json_cached(DATA_PATH)
CAL     = load_json_cached(CAL_PATH)
if not METRICS:
    with st.spinner("Creo metriche di fallback dalla stagione scorsa + corrente (football-data.co.uk)..."):
        METRICS = _metrics_from_fd()

# ---------- Sidebar ----------
st.sidebar.header("Opzioni")
horizon   = st.sidebar.number_input("Giorni futuri", 1, 60, 30, 1)
kick_hour = st.sidebar.slider("Ora indicativa (meteo)", 12, 21, 18)
use_meteo = st.sidebar.checkbox("Meteo automatico", value=True)
use_fd    = st.sidebar.checkbox("Usa football-data.org (API)", value=False)
use_local_cache = st.sidebar.checkbox("Usa cache locale (fallback)", value=True)
use_worldfootball = st.sidebar.checkbox("Usa WorldFootball (consigliato)", value=True)

if st.sidebar.button("🔄 Forza aggiornamento dati"):
    with st.spinner("Rigenero storico e calibratori..."):
        ok = run_full_update()
    if ok:
        st.success("Aggiornato! Premi Rerun in alto a destra.")

# ---------- Recupera partite + diagnostica ----------
fixtures, diag_df = fetch_all_fixtures(horizon, use_fd, use_local_cache, use_worldfootball)
fixtures, diag_df = _sticky_merge(fixtures, diag_df)

with st.expander("🔍 Diagnostica fonti partite"):
    st.dataframe(diag_df, use_container_width=True)

# ---------- Calcolatore personalizzato ----------
st.subheader("🧮 Calcolatore personalizzato (tiri singola squadra + angoli totali)")

with st.expander("Incolla partite (es. 'Torino - Fiorentina'), una per riga"):
    manual_date = st.text_input("Data (YYYY-MM-DD) per le righe incollate", value=str(date.today()))
    manual_text = st.text_area("Partite", value="", height=100)
    added = 0
    if manual_text.strip():
        for line in manual_text.splitlines():
            m = re.match(r"^\s*(.+?)\s*[-–]\s*(.+?)\s*$", line)
            if m:
                fixtures.append({"code":"MAN","league":"Manuale","date":manual_date,"home":m.group(1).strip(),"away":m.group(2).strip(),"source":"Manuale"})
                added += 1
        if added:
            st.success(f"Aggiunte {added} partite manuali.")

if not fixtures:
    st.info("Nessuna partita trovata nell'orizzonte selezionato. Aumenta 'Giorni futuri' nella sidebar.")
else:
    labels = [f"{f['league']} • {f['date']} • {f['home']} - {f['away']}  ({f['source']})" for f in fixtures]
    pick = st.selectbox("Scegli una partita", labels, index=0)
    fx = fixtures[labels.index(pick)]
    code, home, away, date_iso = fx["code"], fx["home"], fx["away"], fx["date"]
    tz = LEAGUE_TZ.get(code, "Europe/Rome")

    c1, c2 = st.columns(2)
    with c1:
        th_home = st.number_input(f"Soglia tiri {home} (usa .5, es. 10.5)", min_value=0.0, step=0.5, value=10.5)
    with c2:
        th_away = st.number_input(f"Soglia tiri {away} (usa .5, es. 12.5)", min_value=0.0, step=0.5, value=12.5)

    # Meteo
    rain=snow=wind=hot=cold=False
    if use_meteo:
        latlon = geocode_team_fallback(home, code, autosave=True)
        if latlon and latlon.get("lat") and latlon.get("lon"):
            wx = fetch_openmeteo_conditions(latlon["lat"], latlon["lon"], date_iso, hour_local=kick_hour, tz=tz) or {}
            rain = wx.get('rain', False); snow = wx.get('snow', False)
            wind = wx.get('wind_strong', False); hot = wx.get('hot', False); cold = wx.get('cold', False)
        else:
            wx = {}

    # Parametri
    METRICS = METRICS or {}
    CAL     = CAL or {}
    shots_params   = compute_lambda_and_var(METRICS, code, home, away, "tiri")
    corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")
    if not shots_params:
        st.warning("Dati squadra non trovati in METRICS per questa partita.")
    else:
        lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
        if corners_params:
            lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params
        else:
            lam_h_c=lam_a_c=vr_h_c=vr_a_c=None

        lam_h_s = adjust_for_weather(lam_h_s, 'tiri', (rain, snow, wind, hot, cold))
        lam_a_s = adjust_for_weather(lam_a_s, 'tiri', (rain, snow, wind, hot, cold))
        if lam_h_c is not None:
            lam_h_c = adjust_for_weather(lam_h_c, 'angoli', (rain, snow, wind, hot, cold))
        if lam_a_c is not None:
            lam_a_c = adjust_for_weather(lam_a_c, 'angoli', (rain, snow, wind, hot, cold))

        def k_from_half(th): return (int(th) + 1) if ((th % 1) == 0.5) else int(np.floor(th)) + 1
        kH = k_from_half(th_home)
        pH = finalize_probability(kH, lam_h_s, var_factor=vr_h_s, prefer_nb=True)
        pH = apply_isotonic(CAL, code, "tiri", "home", kH, pH)
        oh, uh = round(pH*100,1), round((1.0-pH)*100,1)

        kA = k_from_half(th_away)
        pA = finalize_probability(kA, lam_a_s, var_factor=vr_a_s, prefer_nb=True)
        pA = apply_isotonic(CAL, code, "tiri", "away", kA, pA)
        oa, ua = round(pA*100,1), round((1.0-pA)*100,1)

        st.markdown(f"**Probabilità tiri – {home}**: Over {th_home} ➜ **{oh}%**, Under {th_home} ➜ **{uh}%**")
        st.markdown(f"**Probabilità tiri – {away}**: Over {th_away} ➜ **{oa}%**, Under {th_away} ➜ **{ua}%**")

        # Angoli totali over vari
        st.markdown("**Angoli totali – Over automatici**")
        if lam_h_c is None or lam_a_c is None or lam_h_c<=0 or lam_a_c<=0:
            st.info("Dati corner incompleti: mostro solo tiri.")
        else:
            lam_tot = float(lam_h_c + lam_a_c)
            var_tot = float(lam_h_c*vr_h_c + lam_a_c*vr_a_c)
            vf_tot  = (var_tot/lam_tot) if lam_tot > 0 else 1.0
            rows_ct = []
            for th in [5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5]:
                k_tot = int(np.floor(th)) + 1
                p_over = finalize_probability(k_tot, lam_tot, var_factor=vf_tot, prefer_nb=True)
                p_over = apply_isotonic(CAL, code, "angoli_tot", "total", k_tot, p_over)
                rows_ct.append({"Soglia": f"Over {th}", "Probabilità": f"{round(p_over*100,1)}%"})
            st.dataframe(pd.DataFrame(rows_ct), use_container_width=True)

# ---------- Dashboard veloce ----------
st.subheader("📊 Partite prossimi giorni – soglie standard (veloce)")
fixtures_sorted = sorted(fixtures, key=lambda x: (x["date"], x["league"], x["home"])) if fixtures else []
if not fixtures_sorted:
    st.info("Nessuna partita nell’orizzonte selezionato.")
else:
    rows, only_list = [], []
    for fx in fixtures_sorted:
        code, home, away, date_iso = fx["code"], fx["home"], fx["away"], fx["date"]
        shots_params   = compute_lambda_and_var(METRICS, code, home, away, "tiri")
        corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")
        if not shots_params:
            only_list.append(fx)
            continue
        lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
        row = {"Lega": LEAGUE_NAMES.get(code, code), "Data": date_iso, "Match": f"{home}-{away}"}
        for k in CFG['default_thresholds']['shots']:
            p_h = finalize_probability(int(k), lam_h_s, var_factor=vr_h_s, prefer_nb=True)
            p_a = finalize_probability(int(k), lam_a_s, var_factor=vr_a_s, prefer_nb=True)
            p_h = apply_isotonic(CAL, code, "tiri", "home", int(k), p_h)
            p_a = apply_isotonic(CAL, code, "tiri", "away", int(k), p_a)
            row[f"tiri_H≥{k}"] = round(p_h*100,1)
            row[f"tiri_A≥{k}"] = round(p_a*100,1)

        if corners_params:
            lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params
            for k in CFG['default_thresholds']['corners']:
                p_hc = finalize_probability(int(k), lam_h_c, var_factor=vr_h_c, prefer_nb=True)
                p_ac = finalize_probability(int(k), lam_a_c, var_factor=vr_a_c, prefer_nb=True)
                p_hc = apply_isotonic(CAL, code, "angoli", "home", int(k), p_hc)
                p_ac = apply_isotonic(CAL, code, "angoli", "away", int(k), p_ac)
                row[f"angoli_H≥{k}"] = round(p_hc*100,1)
                row[f"angoli_A≥{k}"] = round(p_ac*100,1)
        rows.append(row)

    if rows:
        df_out = pd.DataFrame(rows)
        try:
            df_out['Data_s'] = pd.to_datetime(df_out['Data'], errors='coerce')
            df_out = df_out.sort_values(['Data_s','Lega','Match']).drop(columns=['Data_s'])
        except Exception:
            pass
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Scarica CSV unico", df_out.to_csv(index=False).encode('utf-8'),
                           "dashboard_prob_calibrate.csv", "text/csv")

    if only_list:
        st.markdown("**Partite senza dati completi (mostrate comunque):**")
        st.write(pd.DataFrame(only_list)[['league','date','home','away','source']])

st.caption("Fonti: FPL (PL), WorldFootball (Serie A/La Liga/Bundes), football-data.org (opzionale), cache locale. Meteo: Open-Meteo. Fallback completo se 'pipeline' assente.")
