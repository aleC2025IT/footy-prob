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

def auto_update_if_stale(max_age_hours=18):
    stale_metrics = data_age_hours(DATA_PATH) > max_age_hours
    stale_cal = data_age_hours(CAL_PATH) > max_age_hours
    if stale_metrics or stale_cal:
        with st.spinner("Aggiornamento automatico dei dati in corso..."):
            r1 = subprocess.run([sys.executable, "-m", "pipeline.export_json"], cwd=ROOT, capture_output=True, text=True)
            r2 = subprocess.run([sys.executable, "-m", "pipeline.build_calibrators"], cwd=ROOT, capture_output=True, text=True)
            if r1.returncode != 0:
                st.error("Errore metriche: " + (r1.stderr or r1.stdout)[-400:])
            if r2.returncode != 0:
                st.error("Errore calibratori: " + (r2.stderr or r2.stdout)[-400:])
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
    """
    metric: "tiri" or "angoli" or "angoli_tot"
    side_key: "home" | "away" | "total"
    k_int: integer threshold (e.g., 11 for over 10.5)
    """
    # map metric
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
league_map = {"I1": "Serie A", "E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga"}

# ---- Controls ----
st.sidebar.header("Opzioni")
horizon = st.sidebar.number_input("Giorni futuri", 1, 14, 7, 1)
kick_hour = st.sidebar.slider("Ora indicativa (meteo)", 12, 21, 18)
use_meteo = st.sidebar.checkbox("Meteo automatico", value=True)
default_shots = CFG['default_thresholds']['shots']
default_corners = CFG['default_thresholds']['corners']

# ---- Build fixtures list (for dashboard & picker) ----
fixtures = []
rows_dashboard = []
for code in ["I1", "E0", "SP1", "D1"]:
    df_fix = get_upcoming_fixtures(code, days=horizon)
    if df_fix.empty:
        continue
    tz = LEAGUE_TZ.get(code, "Europe/Rome")
    for _, r in df_fix.iterrows():
        home, away, date_iso = r['home'], r['away'], str(r['date'])
        if code not in METRICS or home not in METRICS[code]['teams'] or away not in METRICS[code]['teams']:
            continue

        # Meteo flags
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

        # Shots & corners lambdas/var
        lam_h_s, lam_a_s, vr_h_s, vr_a_s = compute_lambda_and_var(METRICS, code, home, away, "tiri")
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = compute_lambda_and_var(METRICS, code, home, away, "angoli")

        lam_h_s = adjust_for_weather(lam_h_s, 'tiri', (rain, snow, wind, hot, cold))
        lam_a_s = adjust_for_weather(lam_a_s, 'tiri', (rain, snow, wind, hot, cold))
        lam_h_c = adjust_for_weather(lam_h_c, 'angoli', (rain, snow, wind, hot, cold))
        lam_a_c = adjust_for_weather(lam_a_c, 'angoli', (rain, snow, wind, hot, cold))

        # Dashboard default probabilities (calibrate)
        def pack(metric_name, lam_h, vr_h, lam_a, vr_a, ths):
            out = {}
            for k in ths:
                p_h = finalize_probability(int(k), lam_h, var_factor=vr_h, prefer_nb=True)
                p_a = finalize_probability(int(k), lam_a, var_factor=vr_a, prefer_nb=True)
                p_h_c = apply_isotonic(CAL, code, metric_name, "home", int(k), p_h)
                p_a_c = apply_isotonic(CAL, code, metric_name, "away", int(k), p_a)
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
        rows_dashboard.append(row)

        fixtures.append({
            "label": f"{league_map[code]} • {date_iso} • {home} - {away}",
            "code": code, "date": date_iso,
            "home": home, "away": away,
            "tz": tz,
            "lam_s": (lam_h_s, lam_a_s), "vr_s": (vr_h_s, vr_a_s),
            "lam_c": (lam_h_c, lam_a_c), "vr_c": (vr_h_c, vr_a_c),
        })

# ---- Calcolatore personalizzato (soglie libere per tiri, tabella automatica per angoli totali) ----
st.subheader("🧮 Calcolatore personalizzato (tiri singola squadra + angoli totali)")

if not fixtures:
    st.info("Nessuna partita trovata adesso. Appena ci sono partite nei prossimi giorni, qui comparirà l’elenco.")
else:
    labels = [f["label"] for f in fixtures]
    pick = st.selectbox("Scegli una partita", labels, index=0)
    fx = fixtures[labels.index(pick)]
    code, home, away = fx["code"], fx["home"], fx["away"]

    c1, c2 = st.columns(2)
    with c1:
        th_home = st.number_input(f"Soglia tiri {home} (usa .5, es. 10.5)", min_value=0.0, step=0.5, value=10.5)
    with c2:
        th_away = st.number_input(f"Soglia tiri {away} (usa .5, es. 12.5)", min_value=0.0, step=0.5, value=12.5)

    lam_h_s, lam_a_s = fx["lam_s"]
    vr_h_s, vr_a_s = fx["vr_s"]
    lam_h_c, lam_a_c = fx["lam_c"]
    vr_h_c, vr_a_c = fx["vr_c"]

    def team_over_under(prob_over):
        prob_over = float(prob_over)
        prob_under = 1.0 - prob_over
        return round(prob_over*100,1), round(prob_under*100,1)

    # Over/Under per tiri HOME
    kH = int(th_home) + 1 if abs(th_home - int(th_home) - 0.5) < 1e-9 or th_home % 1 == 0.5 else int(np.floor(th_home)) + 1
    pH_raw = finalize_probability(kH, lam_h_s, var_factor=vr_h_s, prefer_nb=True)
    pH_cal = apply_isotonic(CAL, code, "tiri", "home", kH, pH_raw)
    oh, uh = team_over_under(pH_cal)

    # Over/Under per tiri AWAY
    kA = int(th_away) + 1 if abs(th_away - int(th_away) - 0.5) < 1e-9 or th_away % 1 == 0.5 else int(np.floor(th_away)) + 1
    pA_raw = finalize_probability(kA, lam_a_s, var_factor=vr_a_s, prefer_nb=True)
    pA_cal = apply_isotonic(CAL, code, "tiri", "away", kA, pA_raw)
    oa, ua = team_over_under(pA_cal)

    st.markdown(f"**Probabilità tiri** – {home}: **Over {th_home} ➜ {oh}%** • **Under {th_home} ➜ {uh}%**")
    st.markdown(f"**Probabilità tiri** – {away}: **Over {th_away} ➜ {oa}%** • **Under {th_away} ➜ {ua}%**")

    # Angoli totali: lista automatica Over 5.5, 6.5, ...
    st.markdown("**Angoli totali – Over automatici**")
    thresholds_half = [x + 0.5 for x in range(5, 17)]  # 5.5 ... 16.5
    table = []
    # mean/var total (assumendo indipendenza)
    var_h = lam_h_c * vr_h_c
    var_a = lam_a_c * vr_a_c
    lam_tot = lam_h_c + lam_a_c
    var_tot = var_h + var_a
    var_factor_tot = (var_tot / lam_tot) if lam_tot > 0 else 1.0

    for th in thresholds_half:
        k_tot = int(np.floor(th)) + 1  # es: 5.5 -> 6
        p_over_raw = finalize_probability(k_tot, lam_tot, var_factor=var_factor_tot, prefer_nb=True)
        # calibrazione se disponibile (metric=angoli_tot -> corners_total/side=total)
        p_over_cal = apply_isotonic(CAL, code, "angoli_tot", "total", k_tot, p_over_raw)
        table.append({"Soglia": f"Over {th}", "Probabilità": f"{round(p_over_cal*100,1)}%"})

    st.dataframe(pd.DataFrame(table), use_container_width=True)

# ---- Dashboard sintetica con soglie standard (come prima) ----
st.subheader("📊 Partite prossimi giorni – soglie standard (veloce)")
if not rows_dashboard:
    st.warning("Nessuna partita trovata o dati non ancora calcolati. Se è il primo avvio, attendi l'auto-update e poi ricarica.")
else:
    df_out = pd.DataFrame(rows_dashboard)
    try:
        df_out['Data_s'] = pd.to_datetime(df_out['Data'], errors='coerce')
        df_out = df_out.sort_values(['Data_s', 'Lega', 'Match']).drop(columns=['Data_s'])
    except Exception:
        pass
    st.dataframe(df_out, use_container_width=True)
    st.download_button("Scarica CSV unico", df_out.to_csv(index=False).encode('utf-8'),
                       "dashboard_prob_calibrate.csv", "text/csv")

st.caption("Fonti: FBref (calendari), Open-Meteo (meteo), football-data.co.uk (storico). Aggiornamento automatico se dati >18h.")
