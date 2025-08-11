# --- make repo root importable (so we can import pipeline/*) ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# stdlib
import json, datetime
from datetime import timedelta
from pathlib import Path

# third-party
import pandas as pd
import numpy as np
import requests
import streamlit as st
import yaml

# local modeling
from pipeline.modeling.prob_model import combine_strengths, finalize_probability, blended_var_factor
from pipeline.utils.auto_weather import fetch_openmeteo_conditions, LEAGUE_TZ
from pipeline.utils.geocode import geocode_team_fallback

# ---------- Page ----------
st.set_page_config(page_title="v7 • Probabilità calibrate + Meteo", layout="wide")
st.title("v7 • Probabilità calibrate + Meteo • Dashboard automatica")

# ---------- Paths ----------
DATA_PATH = Path('app/public/data/league_team_metrics.json')
CAL_PATH = Path('app/public/data/calibrators.json')
STADIUMS_PATH = Path('app/public/data/stadiums.json')

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
    if not p.exists(): return 1e9
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
    stale_cal = data_age_hours(CAL_PATH) > max_age_hours
    if stale_metrics or stale_cal:
        with st.spinner("Aggiornamento automatico dei dati in corso..."):
            ok = run_full_update()
            if ok:
                st.success("Aggiornamento completato. Se la tabella non appare, premi Rerun.")

# ---------- Fixtures – 3 fonti in cascata (sempre aggiornato) ----------
LEAGUE_NAMES = {"I1":"Serie A","E0":"Premier League","SP1":"La Liga","D1":"Bundesliga"}
FD_COMP = {"E0":"PL","SP1":"PD","D1":"BL1","I1":"SA"}  # football-data codes

def _safe_df():
    return pd.DataFrame(columns=["date","home","away"])

def _fbref_fixtures(code, days):
    # import locale solo qui per evitare dipendenza circolare
    from pipeline.data_sources.fbref_schedule import get_upcoming_fixtures as fbref_get
    try:
        df = fbref_get(code, days=int(days))
        # normalizza
        if not df.empty:
            df["date"] = df["date"].astype(str)
            df["home"] = df["home"].astype(str).str.strip()
            df["away"] = df["away"].astype(str).str.strip()
        return df
    except Exception:
        return _safe_df()

def _fpl_fixtures_premier(days: int) -> pd.DataFrame:
    try:
        teams = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=20).json().get('teams', [])
        id2name = {t['id']: t['name'] for t in teams}
        fx = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=30).json()
        if not isinstance(fx, list): return _safe_df()
        today = datetime.date.today()
        horizon = today + timedelta(days=max(1,int(days)))
        rows = []
        for f in fx:
            kt = f.get('kickoff_time')
            if not kt: continue
            try:
                # kickoff UTC -> date locale UK
                dt_utc = datetime.datetime.fromisoformat(kt.replace('Z','+00:00'))
                d = (dt_utc + (datetime.datetime.now() - datetime.datetime.utcnow())).date()  # approx local date
            except Exception:
                continue
            if d < today or d > horizon: continue
            h, a = id2name.get(f.get('team_h')), id2name.get(f.get('team_a'))
            if h and a:
                rows.append({"date": str(d), "home": h.strip(), "away": a.strip()})
        return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else _safe_df()
    except Exception:
        return _safe_df()

def _football_data_fixtures(code: str, days: int, api_key: str) -> pd.DataFrame:
    comp = FD_COMP.get(code.upper())
    if not comp or not api_key: return _safe_df()
    try:
        tz_name = LEAGUE_TZ.get(code, "Europe/Rome")
        today = datetime.date.today()
        horizon = today + timedelta(days=max(1,int(days)))
        params = {"dateFrom": str(today), "dateTo": str(horizon), "competitions": comp}
        headers = {"X-Auth-Token": api_key}
        r = requests.get("https://api.football-data.org/v4/matches", params=params, headers=headers, timeout=30)
        r.raise_for_status()
        js = r.json()
        rows = []
        for m in js.get("matches", []):
            utc_str = m.get("utcDate")
            if not utc_str: continue
            try:
                dt = datetime.datetime.fromisoformat(utc_str.replace("Z","+00:00"))
                d = dt.date()
            except Exception:
                continue
            if d < today or d > horizon: continue
            h = m.get("homeTeam",{}).get("name")
            a = m.get("awayTeam",{}).get("name")
            if h and a:
                rows.append({"date": str(d), "home": h.strip(), "away": a.strip()})
        return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True) if rows else _safe_df()
    except Exception:
        return _safe_df()

def get_all_fixtures(days:int=21):
    """Ritorna lista partite e da quale fonte sono arrivate per ogni lega."""
    out = []
    diagnostics = []
    for code in ["I1","E0","SP1","D1"]:
        src_used = None
        df = _safe_df()
        if code == "E0":
            df = _fpl_fixtures_premier(days)
            if not df.empty: src_used = "FPL"
        if (df.empty) and API_KEY:
            df = _football_data_fixtures(code, days, API_KEY)
            if not df.empty: src_used = "football-data"
        if df.empty:
            df = _fbref_fixtures(code, days)
            if not df.empty: src_used = "FBref"
        diagnostics.append({"Lega": LEAGUE_NAMES[code], "Fonte": src_used or "—", "Partite": len(df)})
        if not df.empty:
            for _, r in df.iterrows():
                out.append({
                    "code": code,
                    "league": LEAGUE_NAMES[code],
                    "date": str(r["date"]),
                    "home": str(r["home"]).strip(),
                    "away": str(r["away"]).strip(),
                    "source": src_used or "—"
                })
    out = sorted(out, key=lambda x: (x["date"], x["league"], x["home"]))
    return out, pd.DataFrame(diagnostics)

# ---------- Modeling helpers ----------
def compute_lambda_and_var(METRICS, code, home, away, metric):
    if code not in METRICS: return None
    if home not in METRICS[code]['teams'] or away not in METRICS[code]['teams']:
        return None
    lg = METRICS[code]
    league_means = lg['league_means']; league_vr = lg.get('league_var_ratio', {})
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
    j = bisect.bisect_right(xs, p) - 1; j = max(0, min(j, len(xs)-2))
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
with open('settings.yaml','r',encoding='utf-8') as f:
    CFG = yaml.safe_load(f)
auto_update_if_stale(CFG.get('staleness_hours', 18))
METRICS = load_json_cached(DATA_PATH)
CAL = load_json_cached(CAL_PATH)

# ---------- Sidebar ----------
st.sidebar.header("Opzioni")
horizon = st.sidebar.number_input("Giorni futuri", 1, 45, 30, 1)
kick_hour = st.sidebar.slider("Ora indicativa (meteo)", 12, 21, 18)
use_meteo = st.sidebar.checkbox("Meteo automatico", value=True)
if st.sidebar.button("🔄 Forza aggiornamento dati"):
    with st.spinner("Rigenero storico e calibratori..."):
        ok = run_full_update()
    if ok: st.success("Aggiornato! Premi Rerun in alto a destra.")
st.sidebar.caption(f"API football-data: {'✅' if API_KEY else '❌ (opzionale)'}")

# ---------- Recupera TUTTE le partite + Diagnostica ----------
fixtures, diag_df = get_all_fixtures(days=horizon)
with st.expander("🔍 Diagnostica fonti partite"):
    st.write(diag_df)

# ---------- Calcolatore personalizzato ----------
st.subheader("🧮 Calcolatore personalizzato (tiri singola squadra + angoli totali)")

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
    rain = snow = wind = hot = cold = False
    meta = None
    if use_meteo:
        lat, lon, _ = geocode_team_fallback(home, code, autosave=True)
        if lat and lon:
            wx = fetch_openmeteo_conditions(lat, lon, date_iso, hour_local=kick_hour, tz=tz) or {}
            rain = wx.get('rain', False); snow = wx.get('snow', False)
            wind = wx.get('wind_strong', False); hot = wx.get('hot', False); cold = wx.get('cold', False)
            meta = wx.get('meta', {})

    # Parametri se disponibili
    shots_params = compute_lambda_and_var(METRICS, code, home, away, "tiri")
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
        def team_over_under(p_over):
            return round(p_over*100,1), round((1.0-p_over)*100,1)

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
        vf_tot = (var_tot/lam_tot) if lam_tot > 0 else 1.0
        rows_ct = []
        for th in thresholds_half:
            k_tot = int(np.floor(th)) + 1
            p_over = finalize_probability(k_tot, lam_tot, var_factor=vf_tot, prefer_nb=True)
            p_over = apply_isotonic(CAL, code, "angoli_tot", "total", k_tot, p_over)
            rows_ct.append({"Soglia": f"Over {th}", "Probabilità": f"{round(p_over*100,1)}%"})
        st.dataframe(pd.DataFrame(rows_ct), use_container_width=True)

# ---------- Dashboard veloce ----------
st.subheader("📊 Partite prossimi giorni – soglie standard (veloce)")
if not fixtures:
    st.info("Nessuna partita nell’orizzonte selezionato.")
else:
    rows = []
    only_list = []
    for fx in fixtures:
        code, home, away, date_iso = fx["code"], fx["home"], fx["away"], fx["date"]
        shots_params = compute_lambda_and_var(METRICS, code, home, away, "tiri")
        corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")
        if not shots_params or not corners_params:
            only_list.append(fx); continue
        lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params
        row = {"Lega": fx['league'], "Data": date_iso, "Match": f"{home}-{away}"}
        def pack(metric_name, lam_h, vr_h, lam_a, vr_a, ths):
            out = {}
            for k in CFG['default_thresholds'][ 'shots' if metric_name=='tiri' else 'corners' ]:
                p_h = finalize_probability(int(k), lam_h, var_factor=vr_h, prefer_nb=True)
                p_a = finalize_probability(int(k), lam_a, var_factor=vr_a, prefer_nb=True)
                p_h = apply_isotonic(CAL, code, metric_name, "home", int(k), p_h)
                p_a = apply_isotonic(CAL, code, metric_name, "away", int(k), p_a)
                out[f"{metric_name}_H≥{k}"] = round(p_h*100,1)
                out[f"{metric_name}_A≥{k}"] = round(p_a*100,1)
            return out
        row.update(pack("tiri", lam_h_s, vr_h_s, lam_a_s, vr_a_s, CFG['default_thresholds']['shots']))
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

st.caption("Fonti: FPL (PL), football-data.org (se chiave), FBref (fallback). Meteo: Open-Meteo. Aggiornamento automatico se dati >18h.")
