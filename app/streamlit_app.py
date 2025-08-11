# --- make repo root importable (so we can import pipeline/*) ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# stdlib
import json, datetime, bisect, subprocess
from pathlib import Path

# third-party
import pandas as pd
import numpy as np
import streamlit as st
import yaml

# local modules
from pipeline.modeling.prob_model import combine_strengths, finalize_probability, blended_var_factor
from pipeline.data_sources.fbref_schedule import get_upcoming_fixtures
from pipeline.utils.auto_weather import fetch_openmeteo_conditions, LEAGUE_TZ
from pipeline.utils.geocode import geocode_team_fallback

# ---- Page config ----
st.set_page_config(page_title="v7 • Probabilità calibrate + Meteo", layout="wide")
st.title("v7 • Probabilità calibrate + Meteo • Dashboard automatica")

# ---- Paths ----
DATA_PATH = Path('app/public/data/league_team_metrics.json')
CAL_PATH = Path('app/public/data/calibrators.json')
STADIUMS_PATH = Path('app/public/data/stadiums.json')

# ---- Helpers ----
@st.cache_data(show_spinner=False)
def load_json_cached(path: str):
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

def auto_update_if_stale(max_age_hours=18):
    stale_metrics = data_age_hours(DATA_PATH) > max_age_hours
    stale_cal = data_age_hours(CAL_PATH) > max_age_hours
    if stale_metrics or stale_cal:
        with st.spinner("Aggiornamento automatico dei dati in corso..."):
            r1 = subprocess.run([sys.executable, "pipeline/export_json.py"], capture_output=True, text=True)
            r2 = subprocess.run([sys.executable, "pipeline/build_calibrators.py"], capture_output=True, text=True)
            if r1.returncode != 0:
                st.error("Errore metriche: " + (r1.stderr or r1.stdout)[-300:])
            if r2.returncode != 0:
                st.error("Errore calibratori: " + (r2.stderr or r2.stdout)[-300:])
            st.cache_data.clear()
            st.success("Aggiornamento completato. Ricarica se la tabella non appare.")

def ensure_coords(league_code, team):
    STADIUMS = load_json_cached(STADIUMS_PATH)
    league_map_local = STADIUMS.get(league_code, {})
    if team in league_map_local and 'lat' in league_map_local[team] and 'lon' in league_map_local[team]:
        return league_map_local[team]['lat'], league_map_local[team]['lon'], "mapping"
    res = geocode_team_fallback(team, league_code, autosave=True)
    if res and res.get('lat') and res.get('lon'):
        st.cache_data.clear()  # stadiums.json changed
        return res['lat'], res['lon'], "geocoded"
    return None, None, None

def compute_lambda_and_var(METRICS, code, home, away, metric):
    lg = METRICS[code]
    league_means = lg['league_means']
    league_vr = lg.get('league_var_ratio', {})
    th = lg['teams'][home]
    ta = lg['teams'][away]
    if metric == "tiri":
        team_for_home = th['shots_for_home']; league_for_home = league_means['shots_home_for']
        opp_against_away = ta['shots_against_away']; league_against_away = league_means['shots_away_against']
        league_mean = (league_means['shots_home_for'] + league_means['shots_away_for']) / 2.0
        team_for_away = ta['shots_for_away']; opp_against_home = th['shots_against_home']
        league_against_home = league_means['shots_home_against']
        H_home, H_away = 1.05, 0.95
        vr_home = blended_var_factor(th.get('vr_shots_for_home'), ta.get('vr_shots_against_away'),
                                     league_vr.get('shots_home_for', 1.1), floor=1.0, ceil=2.0)
        vr_away = blended_var_factor(ta.get('vr_shots_for_away'), th.get('vr_shots_against_home'),
                                     league_vr.get('shots_away_for', 1.1), floor=1.0, ceil=2.0)
    else:
        team_for_home = th['corners_for_home']; league_for_home = league_means['corners_home_for']
        opp_against_away = ta['corners_against_away']; league_against_away = league_means['corners_away_against']
        league_mean = (league_means['corners_home_for'] + league_means['corners_away_for']) / 2.0
        team_for_away = ta['corners_for_away']; opp_against_home = th['corners_against_home']
        league_against_home = league_means['corners_home_against']
        H_home, H_away = 1.03, 0.97
        vr_home = blended_var_factor(th.get('vr_corners_for_home'), ta.get('vr_corners_against_away'),
                                     league_vr.get('corners_home_for', 1.3), floor=1.1, ceil=2.5)
        vr_away = blended_var_factor(ta.get('vr_corners_for_away'), th.get('vr_corners_against_home'),
                                     league_vr.get('corners_away_for', 1.3), floor=1.1, ceil=2.5)
    lam_home = combine_strengths(team_for_home, league_for_home, opp_against_away, league_against_away, league_mean, H_home)
    lam_away = combine_strengths(team_for_away, league_for_home, opp_against_home, league_against_home, league_mean, H_away)
    return lam_home, lam_away, vr_home, vr_away

def apply_isotonic(CAL, code, metric, is_home, k, p):
    metric_key = "shots" if metric == "tiri" else "corners"
    side_key = "home" if is_home else "away"
    cal = CAL.get(code, {}).get(metric_key, {}).get(side_key, {})
    if not cal:
        return p
    key = str(int(k))
    if key not in cal:
        keys = sorted([int(x) for x in cal.keys()]) if cal else []
        if not keys:
            return p
        key = str(min(keys, key=lambda x: abs(x - int(k))))
    xs, ys = cal[key]['x'], cal[key]['y']
    if p <= xs[0]:
        return ys[0]
    if p >= xs[-1]:
        return ys[-1]
    j = bisect.bisect_right(xs, p) - 1
    j = max(0, min(j, len(xs) - 2))
    x0, x1 = xs[j], xs[j + 1]
    y0, y1 = ys[j], ys[j + 1]
    t = 0 if x1 == x0 else (p - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

# ---- Settings & auto-update ----
with open('settings.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

auto_update_if_stale(CFG.get('staleness_hours', 18))

METRICS = load_json_cached(DATA_PATH)
CAL = load_json_cached(CAL_PATH)
league_map = {"I1": "Serie A", "E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga"}

# ---- UI controls ----
st.sidebar.header("Opzioni")
horizon = st.sidebar.number_input("Giorni futuri", 1, 14, 7, 1)
kick_hour = st.sidebar.slider("Ora indicativa (meteo)", 12, 21, 18)
use_meteo = st.sidebar.checkbox("Meteo automatico", value=True)
default_shots = CFG['default_thresholds']['shots']
default_corners = CFG['default_thresholds']['corners']

# ---- Build dashboard rows ----
rows = []
for code in ["I1", "E0", "SP1", "D1"]:
    df_fix = get_upcoming_fixtures(code, days=horizon)
    if df_fix.empty:
        continue
    tz = LEAGUE_TZ.get(code, "Europe/Rome")
    for _, r in df_fix.iterrows():
        home, away, date_iso = r['home'], r['away'], str(r['date'])
        if code not in METRICS or home not in METRICS[code]['teams'] or away not in METRICS[code]['teams']:
            continue

        lam_h_s, lam_a_s, vr_h_s, vr_a_s = compute_lambda_and_var(METRICS, code, home, away, "tiri")
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = compute_lambda_and_var(METRICS, code, home, away, "angoli")

        rain = snow = wind = hot = cold = False
        meta = None
        if use_meteo:
            lat, lon, _ = ensure_coords(code, home)
            if lat and lon:
                wx = fetch_openmeteo_conditions(lat, lon, date_iso, hour_local=kick_hour, tz=tz) or {}
                rain = wx.get('rain', False)
                snow = wx.get('snow', False)
                wind = wx.get('wind_strong', False)
                hot = wx.get('hot', False)
                cold = wx.get('cold', False)
                meta = wx.get('meta', {})

        def adj(lam, metric_name):
            if rain:
                lam *= 0.97 if metric_name == 'tiri' else 1.06
            if snow:
                lam *= 0.94 if metric_name == 'tiri' else 1.10
            if wind:
                lam *= 0.96 if metric_name == 'tiri' else 1.08
            if hot:
                lam *= 0.98
            if cold:
                lam *= 0.99 if metric_name == 'tiri' else 1.01
            return lam

        lam_h_s = adj(lam_h_s, 'tiri'); lam_a_s = adj(lam_a_s, 'tiri')
        lam_h_c = adj(lam_h_c, 'angoli'); lam_a_c = adj(lam_a_c, 'angoli')

        def pack(metric_name, lam_h, vr_h, lam_a, vr_a, ths):
            out = {}
            for k in ths:
                p_h = finalize_probability(int(k), lam_h, var_factor=vr_h, prefer_nb=True)
                p_a = finalize_probability(int(k), lam_a, var_factor=vr_a, prefer_nb=True)
                p_h_c = apply_isotonic(CAL, code, metric_name, True, int(k), p_h)
                p_a_c = apply_isotonic(CAL, code, metric_name, False, int(k), p_a)
                out[f"{metric_name}_H≥{k}"] = round(p_h_c * 100, 1)
                out[f"{metric_name}_A≥{k}"] = round(p_a_c * 100, 1)
            return out

        probs = {}
        probs.update(pack("tiri", lam_h_s, vr_h_s, lam_a_s, vr_a_s, default_shots))
        probs.update(pack("angoli", lam_h_c, vr_h_c, lam_a_c, vr_a_c, default_corners))

        row = {"Lega": league_map[code], "Data": date_iso, "Match": f"{home}-{away}"}
        row.update(probs)
        if meta:
            row.update({"T°C": meta.get("temp_c"), "Prec(mm)": meta.get("precip_mm"), "Vento(km/h)": meta.get("wind_kmh")})
        rows.append(row)

# ---- Render ----
if not rows:
    st.warning("Nessuna partita trovata o dati non ancora calcolati. Se è il primo avvio, attendi l'auto-update e poi ricarica.")
else:
    df_out = pd.DataFrame(rows)
    try:
        df_out['Data_s'] = pd.to_datetime(df_out['Data'], errors='coerce')
        df_out = df_out.sort_values(['Data_s', 'Lega', 'Match']).drop(columns=['Data_s'])
    except Exception:
        pass
    st.dataframe(df_out, use_container_width=True)
    st.download_button("Scarica CSV unico", df_out.to_csv(index=False).encode('utf-8'),
                       "dashboard_prob_calibrate.csv", "text/csv")

st.caption("Fonti: FBref (calendari), Open-Meteo (meteo), football-data.co.uk (storico). Aggiornamento automatico se dati >18h.")
