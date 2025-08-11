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
    if code not in METRICS: return None
    if home not in METRICS[code]['teams'] or away not in METRICS[code]['teams']:
        return None
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
    else:  # angoli
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

def apply_isotonic(CAL, code, metric, side_key, k_int, p):
    if metric == "tiri":
        metric_key = "shots"
    elif metric == "angoli":
        metric_key = "corners"
    elif metric == "angoli_tot":
        metric_key = "corners_total"
    else:
        return p
    cal = CAL.get(code, {}).get(metric_key, {}).get(side_key, {})
    if not cal:
        return p
    key = str(int(k_int))
    if key not in cal:
        keys = sorted([int(x) for x in cal.keys()]) if cal else []
        if not keys:
            return p
        key = str(min(keys, key=lambda x: abs(x - int(k_int))))
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

def adjust_for_weather(lam, metric_name, flags):
    rain, snow, wind, hot, cold = flags
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

# ---- Settings & auto-update ----
with open('settings.yaml', 'r', encoding='utf-8') as f:
    CFG = yaml.safe_load(f)

auto_update_if_stale(CFG.get('staleness_hours', 18))

METRICS = load_json_cached(DATA_PATH)
CAL = load_json_cached(CAL_PATH)
league_map = {"I1": "Serie A", "E0": "Premier League", "SP1": "La Liga", "D1": "Bun_
