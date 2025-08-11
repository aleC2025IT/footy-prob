# --- make repo root importable ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# stdlib
import json, re, time, datetime
from datetime import timedelta
from pathlib import Path

# third-party
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yaml

# local modeling / utils
from pipeline.modeling.prob_model import combine_strengths, finalize_probability, blended_var_factor
from pipeline.utils.auto_weather import fetch_openmeteo_conditions, LEAGUE_TZ
from pipeline.utils.geocode import geocode_team_fallback

# ---------- Page ----------
st.set_page_config(page_title="v7 • Probabilità calibrate + Meteo", layout="wide")
st.title("v7 • Probabilità calibrate + Meteo • Dashboard automatica")

# ---------- Paths ----------
DATA_PATH = Path('app/public/data/league_team_metrics.json')
CAL_PATH  = Path('app/public/data/calibrators.json')

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

def _safe_df():
    return pd.DataFrame(columns=["date","home","away"])

# ------- FBref helpers (robusti: UA + retry + cache 15') -------
FBREF_BASE = {
    "I1":  "https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures",
    "E0":  "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "SP1": "https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures",
    "D1":  "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures",
}

def _season_slug_today():
    today = datetime.date.today()
    start_year = today.year if today.month >= 7 else (today.year - 1)
    return f"{start_year}-{start_year+1}"

def _ua_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36")
    })
    return s

def _fbref_pick_current_url(code: str, sess: requests.Session) -> str:
    base = FBREF_BASE[code]
    try:
        html = sess.get(base, timeout=25).text
        season = _season_slug_today()
        m = re.search(r'href="(/en/comps/\d+/\d{4}-\d{4}/schedule/[^"]*Scores-and-Fixtures[^"]*)"', html)
        if m and season in m.group(1):
            return "https://fbref.com" + m.group(1)
        return base
    except Exception:
        return base

def _get_html_retry(sess: requests.Session, url: str, tries: int = 3) -> str:
    last = ""
    for i in range(tries):
        r = sess.get(url, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "3"))
            time.sleep(min(wait, 5))
            last = "HTTP 429"
            continue
        if r.status_code >= 400:
            last = f"HTTP {r.status_code}"
            if i < tries - 1:
                time.sleep(2)
                continue
            r.raise_for_status()
        r.raise_for_status()
        return r.text
    raise RuntimeError(last or "HTTP error")

def _fbref_fixtures_aggressive(code, days):
    """Legge la tabella fixture FBref della STAGIONE CORRENTE, filtrando l’orizzonte (robusto)."""
    try:
        sess = _ua_session()
        url  = _fbref_pick_current_url(code, sess)
        html = _get_html_retry(sess, url, tries=3)
        dfs = pd.read_html(html)
        target = None
        for df in dfs:
            cols = [str(c).strip().lower() for c in df.columns]
            if 'date' in cols and any('home' in c for c in cols) and any('away' in c for c in cols):
                target = df
                break
        if target is None:
            return _safe_df(), "FBref: tabella non trovata"

        df = target.copy()
        ren = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc == 'date': ren[c] = 'date'
            elif 'home' in lc: ren[c] = 'home'
            elif 'away' in lc: ren[c] = 'away'
        df = df.rename(columns=ren)
        df = df[[c for c in ['date','home','away'] if c in df.columns]].dropna(subset=['date','home','away'])
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        today = datetime.date.today()
        horizon = today + timedelta(days=max(1, int(days)))
        df = df[(df['date'] >= today) & (df['date'] <= horizon)]
        df['home'] = df['home'].astype(str).str.strip()
        df['away'] = df['away'].astype(str).str.strip()
        return df[['date','home','away']].drop_duplicates().reset_index(drop=True), ""
    except requests.HTTPError as e:
        code_msg = getattr(e, 'response', None) and e.response.status_code
        return _safe_df(), f"FBref HTTP {code_msg}"
    except Exception as e:
        return _safe_df(), f"FBref err: {type(e).__name__}"

@st.cache_data(ttl=900, show_spinner=False)  # 15 minuti
def _fbref_fixtures_aggressive_cached(code, days):
    return _fbref_fixtures_aggressive(code, days)

# ------- FPL / football-data (con cache 10') -------
@st.cache_data(ttl=600, show_spinner=False)
def _fpl_fixtures_premier(days):
    try:
        teams = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json().get('teams', [])
        id2name = {t['id']: t['name'] for t in teams}
        fx = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=30).json()
        if not isinstance(fx, list):
            return _safe_df(), "FPL: risposta inattesa"
        today = datetime.date.today()
        horizon = today + timedelta(days=max(1, int(days)))
        rows = []
        for f in fx:
            kt = f.get('kickoff_time')
            if not kt:
                continue
            try:
                dt_utc = datetime.datetime.fromisoformat(kt.replace('Z', '+00:00'))
                d = (dt_utc + (datetime.datetime.now() - datetime.datetime.utcnow())).date()
            except Exception:
                continue
            if d < today or d > horizon:
                continue
            h = id2name.get(f.get('team_h')); a = id2name.get(f.get('team_a'))
            if h and a:
                rows.append({"date": str(d), "home": h.strip(), "away": a.strip()})
        return (pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else _safe_df(), "")
    except Exception as e:
        return _safe_df(), f"FPL err: {type(e).__name__}"

@st.cache_data(ttl=600, show_spinner=False)
def _football_data_fixtures(code, days, api_key):
    comp = {"E0":"PL","SP1":"PD","D1":"BL1","I1":"SA"}.get(code.upper())
    if not comp or not api_key:
        return _safe_df(), "no key/comp"
    try:
        today = datetime.date.today()
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
            if not utc_str:
                continue
            try:
                d = datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00")).date()
            except Exception:
                continue
            if d < today or d > horizon:
                continue
            h = m.get("homeTeam", {}).get("name"); a = m.get("awayTeam", {}).get("name")
            if h and a:
                rows.append({"date": str(d), "home": h.strip(), "away": a.strip()})
        return (pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else _safe_df(), "")
    except Exception as e:
        return _safe_df(), f"football-data err: {type(e).__name__}"

# -------- Sticky cache: non far sparire La Liga se FBref risponde 429 --------
def _sticky_merge(current_fixtures, current_diag):
    prev_fx  = st.session_state.get("last_fixtures", [])
    prev_diag = st.session_state.get("last_diag", None)
    try:
        row = current_diag[current_diag["Lega"]=="La Liga"].iloc[0]
        partite = int(row["Partite"])
        errore  = str(row["Errore"])
    except Exception:
        partite, errore = 0, ""
    if partite == 0 and ("429" in errore or "vuoto" in errore):
        prev_sp1 = [f for f in prev_fx if f.get("code")=="SP1"]
        if prev_sp1:
            cur_no_sp1 = [f for f in current_fixtures if f.get("code")!="SP1"]
            merged = cur_no_sp1 + prev_sp1
            current_fixtures = sorted(merged, key=lambda x: (x["date"], x["league"], x["home"]))
            try:
                idx = current_diag.index[current_diag["Lega"]=="La Liga"][0]
                current_diag.at[idx, "Fonte"] = "cache (FBref)"
                current_diag.at[idx, "Partite"] = len(prev_sp1)
                current_diag.at[idx, "Errore"] = "FBref 429 → cache 15′"
            except Exception:
                pass
    st.session_state["last_fixtures"] = current_fixtures
    st.session_state["last_diag"] = current_diag
    return current_fixtures, current_diag

# ---------- Funzione principale fixtures ----------
def get_all_fixtures(days=30, use_fd=True, force_fbref_sp1=True):
    out, diags = [], []
    for code in ["I1","E0","SP1","D1"]:
        df = _safe_df(); src=None; err=""

        if code=="SP1" and force_fbref_sp1:
            d3, err3 = _fbref_fixtures_aggressive_cached(code, days)
            if not d3.empty: df, src, err = d3, "FBref", ""
            else: err = err3 or "FBref vuoto"
        else:
            # 1) Premier: FPL
            if code=="E0":
                d1, err1 = _fpl_fixtures_premier(days)
                if not d1.empty: df, src = d1, "FPL"
                else: err = err1
            # 2) football-data (se attivo)
            if df.empty and use_fd and API_KEY:
                d2, err2 = _football_data_fixtures(code, days, API_KEY)
                if not d2.empty: df, src, err = d2, "football-data", ""
                else: err = (err + " | " + err2).strip(" |")
            # 3) FBref
            if df.empty:
                d3, err3 = _fbref_fixtures_aggressive_cached(code, days)
                if not d3.empty: df, src, err = d3, "FBref", ""
                else: err = (err + " | " + err3).strip(" |")

        diags.append({"Lega": LEAGUE_NAMES[code], "Fonte": src or "—", "Partite": len(df), "Errore": err or "—"})
        if not df.empty:
            for _, r in df.iterrows():
                out.append({"code": code, "league": LEAGUE_NAMES[code], "date": str(r["date"]),
                            "home": str(r["home"]).strip(), "away": str(r["away"]).strip(), "source": src or "—"})
    out = sorted(out, key=lambda x: (x["date"], x["league"], x["home"]))
    return out, pd.DataFrame(diags)

@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_fixtures(days, use_fd, force_fbref_sp1):
    return get_all_fixtures(days, use_fd, force_fbref_sp1)

# ---------- Modeling helpers ----------
def compute_lambda_and_var(METRICS, code, home, away, metric):
    if code not in METRICS: return None
    if home not in METRICS[code]['teams'] or away not in METRICS[code]['teams']: return None
    lg = METRICS[code]; league_means = lg['league_means']; league_vr = lg.get('league_var_ratio', {})
    th = lg['teams'][home]; ta = lg['teams'][away]
    if metric == "tiri":
        team_for_home = th['shots_for_home']; league_for_home = league_means['shots_home_for']
        opp_against_away = ta['shots_against_away']; league_against_away = league_means['shots_away_against']
        league_mean = (league_means['shots_home_for'] + league_means['shots_away_for'])/2.0
        team_for_away = ta['shots_for_away']; opp_against_home = th['shots_against_home']
        league_against_home = league_means['shots_home_against']
        H_home, H_away = 1.05, 0.95
        vr_home = blended_var_factor(th.get('vr_shots_for_home'), ta.get('vr_shots_against_away'), league_vr.get('shots_home_for', 1.1), 1.0, 2.0)
        vr_away = blended_var_factor(ta.get('vr_shots_for_away'), th.get('vr_shots_against_home'), league_vr.get('shots_away_for', 1.1), 1.0, 2.0)
    else:
        team_for_home = th['corners_for_home']; league_for_home = league_means['corners_home_for']
        opp_against_away = ta['corners_against_away']; league_against_away = league_means['corners_away_against']
        league_mean = (league_means['corners_home_for'] + league_means['corners_away_for'])/2.0
        team_for_away = ta['corners_for_away']; opp_against_home = th['corners_against_home']
        league_against_home = league_means['corners_home_against']
        H_home, H_away = 1.03, 0.97
        vr_home = blended_var_factor(th.get('vr_corners_for_home'), ta.get('vr_corners_against_away'), league_vr.get('corners_home_for', 1.3), 1.1, 2.5)
        vr_away = blended_var_factor(ta.get('vr_corners_for_away'), th.get('vr_corners_against_home'), league_vr.get('corners_away_for', 1.3), 1.1, 2.5)
    lam_home = combine_strengths(team_for_home, league_for_home, opp_against_away, league_against_away, league_mean, H_home)
    lam_away = combine_strengths(team_for_away, league_for_home, opp_against_home, league_against_home, league_mean, H_away)
    return lam_home, lam_away, vr_home, vr_away

def apply_isotonic(CAL, code, metric, side_key, k_int, p):
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
    xs, ys = cal[key]['x'], cal[key]['y']
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
METRICS = load_json_cached(DATA_PATH)
CAL     = load_json_cached(CAL_PATH)

# ---------- Sidebar ----------
st.sidebar.header("Opzioni")
horizon   = st.sidebar.number_input("Giorni futuri", 1, 60, 30, 1)
kick_hour = st.sidebar.slider("Ora indicativa (meteo)", 12, 21, 18)
use_meteo = st.sidebar.checkbox("Meteo automatico", value=True)
use_fd    = st.sidebar.checkbox("Usa football-data.org (API)", value=bool(API_KEY))
force_sp1 = st.sidebar.checkbox("Forza FBref per La Liga", value=True)

if st.sidebar.button("🔄 Forza aggiornamento dati"):
    with st.spinner("Rigenero storico e calibratori..."):
        ok = run_full_update()
    if ok:
        st.success("Aggiornato! Premi Rerun in alto a destra.")
st.sidebar.caption(f"API football-data: {'✅' if API_KEY else '❌ (opzionale)'}")

# ---------- Recupera partite + diagnostica (cache 10′ + sticky) ----------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_all_fixtures(days, use_fd, force_fbref_sp1):
    return get_all_fixtures(days, use_fd, force_fbref_sp1)

fixtures, diag_df = fetch_all_fixtures(horizon, use_fd, force_sp1)
fixtures, diag_df = _sticky_merge(fixtures, diag_df)

with st.expander("🔍 Diagnostica fonti partite"):
    st.write(diag_df)

# ---------- Calcolatore personalizzato ----------
st.subheader("🧮 Calcolatore personalizzato (tiri singola squadra + angoli totali)")

with st.expander("Incolla partite (es. 'Torino - Fiorentina'), una per riga"):
    manual_date = st.text_input("Data (YYYY-MM-DD) per le righe incollate", value=str(datetime.date.today()))
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
    rain=snow=wind=hot=cold=False; meta=None
    if use_meteo:
        latlon = geocode_team_fallback(home, code, autosave=True)
        if latlon and latlon.get("lat") and latlon.get("lon"):
            wx = fetch_openmeteo_conditions(latlon["lat"], latlon["lon"], date_iso, hour_local=kick_hour, tz=tz) or {}
            rain = wx.get('rain', False); snow = wx.get('snow', False)
            wind = wx.get('wind_strong', False); hot = wx.get('hot', False); cold = wx.get('cold', False)
            meta = wx.get('meta', {})

    # Parametri
    METRICS = load_json_cached(DATA_PATH)
    CAL     = load_json_cached(CAL_PATH)
    shots_params   = compute_lambda_and_var(METRICS, code, home, away, "tiri")
    corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")
    if not shots_params or not corners_params:
        st.warning("Dati squadra non ancora pronti per questa partita. Premi 'Forza aggiornamento dati' oppure riprova più tardi.")
    else:
        lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params
        lam_h_s = adjust_for_weather(lam_h_s, 'tiri', (rain, snow, wind, hot, cold))
        lam_a_s = adjust_for_weather(lam_a_s, 'tiri', (rain, snow, wind, hot, cold))
        lam_h_c = adjust_for_weather(lam_h_c, 'angoli', (rain, snow, wind, hot, cold))
        lam_a_c = adjust_for_weather(lam_a_c, 'angoli', (rain, snow, wind, hot, cold))

        def k_from_half(th): return (int(th) + 1) if ((th % 1) == 0.5) else int(np.floor(th)) + 1
        def team_over_under(p_over): return round(p_over*100,1), round((1.0-p_over)*100,1)

        kH = k_from_half(th_home)
        pH = finalize_probability(kH, lam_h_s, var_factor=vr_h_s, prefer_nb=True)
        pH = apply_isotonic(CAL, code, "tiri", "home", kH, pH)
        oh, uh = team_over_under(pH)

        kA = k_from_half(th_away)
        pA = finalize_probability(kA, lam_a_s, var_factor=vr_a_s, prefer_nb=True)
        pA = apply_isotonic(CAL, code, "tiri", "away", kA, pA)
        oa, ua = team_over_under(pA)

        st.markdown(f"**Probabilità tiri – {home}**: Over {th_home} ➜ **{oh}%**, Under {th_home} ➜ **{uh}%**")
        st.markdown(f"**Probabilità tiri – {away}**: Over {th_away} ➜ **{oa}%**, Under {th_away} ➜ **{ua}%**")

        # Angoli totali over 5.5..16.5
        st.markdown("**Angoli totali – Over automatici**")
        thresholds_half = [x + 0.5 for x in range(5, 17)]
        lam_tot = lam_h_c + lam_a_c
        var_tot = lam_h_c*vr_h_c + lam_a_c*vr_a_c
        vf_tot  = (var_tot/lam_tot) if lam_tot > 0 else 1.0
        rows_ct = []
        for th in thresholds_half:
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
    METRICS = load_json_cached(DATA_PATH)
    CAL     = load_json_cached(CAL_PATH)
    rows, only_list = [], []
    for fx in fixtures_sorted:
        code, home, away, date_iso = fx["code"], fx["home"], fx["away"], fx["date"]
        shots_params   = compute_lambda_and_var(METRICS, code, home, away, "tiri")
        corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")
        if not shots_params or not corners_params:
            only_list.append(fx)
            continue

        lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params

        def pack(metric_name, lam_h, vr_h, lam_a, vr_a, ths):
            out = {}
            for k in ths:
                p_h = finalize_probability(int(k), lam_h, var_factor=vr_h, prefer_nb=True)
                p_a = finalize_probability(int(k), lam_a, var_factor=vr_a, prefer_nb=True)
                p_h = apply_isotonic(CAL, code, metric_name, "home", int(k), p_h)
                p_a = apply_isotonic(CAL, code, metric_name, "away", int(k), p_a)
                out[f"{metric_name}_H≥{k}"] = round(p_h*100,1)
                out[f"{metric_name}_A≥{k}"] = round(p_a*100,1)
            return out

        row = {"Lega": LEAGUE_NAMES.get(code, code), "Data": date_iso, "Match": f"{home}-{away}"}
        row.update(pack("tiri",   lam_h_s, vr_h_s, lam_a_s, vr_a_s, CFG['default_thresholds']['shots']))
        row.update(pack("angoli", lam_h_c, vr_h_c, lam_a_c, vr_a_c, CFG['default_thresholds']['corners']))
        rows.append(row)

    if rows:
        df_out = pd.DataFrame(rows)
        try:
            df_out['Data_s'] = pd.to_datetime(df_out['Data'], errors='coerce')
            df_out = df_out.sort_values(['Data_s','Lega','Match']).drop(columns=['Data_s'])
        except Exception:
            pass
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Scarica CSV unico", df_out.to_csv(index=False).encode('utf-8'), "dashboard_prob_calibrate.csv", "text/csv")

    if only_list:
        st.markdown("**Partite senza dati completi (mostrate comunque):**")
        st.write(pd.DataFrame(only_list)[['league','date','home','away','source']])

st.caption("Fonti: FPL (PL), football-data.org (se attivo), FBref (stagione corrente, fallback). Cache 15′ FBref + sticky anti-429 per La Liga. Meteo: Open-Meteo.")
