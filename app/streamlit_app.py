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
league_map = {"I1": "Serie A", "E0": "Premier League", "SP1": "La Liga", "D1": "Bundesliga"}

# ---- Sidebar ----
st.sidebar.header("Opzioni")
horizon = st.sidebar.number_input("Giorni futuri", 1, 31, 21, 1)
kick_hour = st.sidebar.slider("Ora indicativa (meteo)", 12, 21, 18)
use_meteo = st.sidebar.checkbox("Meteo automatico", value=True)
if st.sidebar.button("🔄 Forza aggiornamento dati"):
    with st.spinner("Rigenero storico e calibratori..."):
        ok = run_full_update()
    if ok: st.success("Aggiornato! Premi Rerun in alto a destra.")
default_shots = CFG['default_thresholds']['shots']
default_corners = CFG['default_thresholds']['corners']

# ---- Raccogli TUTTE le partite (mostra anche se mancano metriche) ----
fixtures = []
for code in ["I1", "E0", "SP1", "D1"]:
    df_fix = get_upcoming_fixtures(code, days=horizon)  # già in Europe/Rome
    if df_fix.empty:
        continue
    for _, r in df_fix.iterrows():
        fixtures.append({
            "code": code,
            "league": league_map[code],
            "date": str(r['date']),
            "home": str(r['home']).strip(),
            "away": str(r['away']).strip(),
        })

fixtures = sorted(fixtures, key=lambda x: (x['date'], x['league'], x['home']))

# ---- Calcolatore personalizzato ----
st.subheader("🧮 Calcolatore personalizzato (tiri singola squadra + angoli totali)")

if not fixtures:
    st.info("Nessuna partita trovata nell'orizzonte scelto. Aumenta 'Giorni futuri' nella sidebar.")
else:
    labels = [f"{f['league']} • {f['date']} • {f['home']} - {f['away']}" for f in fixtures]
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
        lat, lon, _ = ensure_coords(code, home)
        if lat and lon:
            wx = fetch_openmeteo_conditions(lat, lon, date_iso, hour_local=kick_hour, tz=tz) or {}
            rain = wx.get('rain', False)
            snow = wx.get('snow', False)
            wind = wx.get('wind_strong', False)
            hot = wx.get('hot', False)
            cold = wx.get('cold', False)
            meta = wx.get('meta', {})

    # LAMBDA/VAR se metriche disponibili
    shots_params = compute_lambda_and_var(METRICS, code, home, away, "tiri")
    corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")

    if not shots_params or not corners_params:
        st.warning("Dati squadra non ancora pronti per questa partita (ma la gara è confermata). Prova a premere 'Forza aggiornamento dati' o riprovare tra poco.")
    else:
        lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params
        lam_h_s = adjust_for_weather(lam_h_s, 'tiri', (rain, snow, wind, hot, cold))
        lam_a_s = adjust_for_weather(lam_a_s, 'tiri', (rain, snow, wind, hot, cold))
        lam_h_c = adjust_for_weather(lam_h_c, 'angoli', (rain, snow, wind, hot, cold))
        lam_a_c = adjust_for_weather(lam_a_c, 'angoli', (rain, snow, wind, hot, cold))

        def k_from_half(th):
            # es: 10.5 -> 11 (P[X>=11])
            if (th % 1) == 0.5:
                return int(th) + 1
            return int(np.floor(th)) + 1

        def team_over_under(p_over):
            p_over = float(p_over)
            return round(p_over*100,1), round((1.0-p_over)*100,1)

        # Tiri home
        kH = k_from_half(th_home)
        pH_raw = finalize_probability(kH, lam_h_s, var_factor=vr_h_s, prefer_nb=True)
        pH_cal = apply_isotonic(CAL, code, "tiri", "home", kH, pH_raw)
        oh, uh = team_over_under(pH_cal)

        # Tiri away
        kA = k_from_half(th_away)
        pA_raw = finalize_probability(kA, lam_a_s, var_factor=vr_a_s, prefer_nb=True)
        pA_cal = apply_isotonic(CAL, code, "tiri", "away", kA, pA_raw)
        oa, ua = team_over_under(pA_cal)

        st.markdown(f"**Probabilità tiri** – {home}: **Over {th_home} ➜ {oh}%** • **Under {th_home} ➜ {uh}%**")
        st.markdown(f"**Probabilità tiri** – {away}: **Over {th_away} ➜ {oa}%** • **Under {th_away} ➜ {ua}%**")

        # Angoli totali – tabella automatica Over 5.5 .. 16.5
        st.markdown("**Angoli totali – Over automatici**")
        thresholds_half = [x + 0.5 for x in range(5, 17)]
        var_h = lam_h_c * vr_h_c
        var_a = lam_a_c * vr_a_c
        lam_tot = lam_h_c + lam_a_c
        var_tot = var_h + var_a
        var_factor_tot = (var_tot / lam_tot) if lam_tot > 0 else 1.0

        rows_ct = []
        for th in thresholds_half:
            k_tot = int(np.floor(th)) + 1
            p_over_raw = finalize_probability(k_tot, lam_tot, var_factor=var_factor_tot, prefer_nb=True)
            p_over_cal = apply_isotonic(CAL, code, "angoli_tot", "total", k_tot, p_over_raw)
            rows_ct.append({"Soglia": f"Over {th}", "Probabilità": f"{round(p_over_cal*100,1)}%"})
        st.dataframe(pd.DataFrame(rows_ct), use_container_width=True)

# ---- Dashboard veloce (soglie standard) ----
st.subheader("📊 Partite prossimi giorni – soglie standard (veloce)")
if not fixtures:
    st.info("Nessuna partita nell’orizzonte selezionato.")
else:
    # Provo a calcolare; se mancano metriche per una gara, la mostro comunque a fondo come 'solo lista'
    rows = []
    only_list = []
    for fx in fixtures:
        code, home, away, date_iso = fx["code"], fx["home"], fx["away"], fx["date"]
        shots_params = compute_lambda_and_var(METRICS, code, home, away, "tiri")
        corners_params = compute_lambda_and_var(METRICS, code, home, away, "angoli")
        if not shots_params or not corners_params:
            only_list.append(fx)
            continue

        lam_h_s, lam_a_s, vr_h_s, vr_a_s = shots_params
        lam_h_c, lam_a_c, vr_h_c, vr_a_c = corners_params
        # Meteo sintetico (no cache per velocità)
        tz = LEAGUE_TZ.get(code, "Europe/Rome")
        lat = lon = None
        if use_meteo:
            latlon = geocode_team_fallback(home, code, autosave=False)
            if latlon:
                lat, lon = latlon.get('lat'), latlon.get('lon')
        T=P=W=None
        if use_meteo and lat and lon:
            wx = fetch_openmeteo_conditions(lat, lon, date_iso, hour_local=kick_hour, tz=tz) or {}
            T = wx.get('meta',{}).get('temp_c'); P = wx.get('meta',{}).get('precip_mm'); W = wx.get('meta',{}).get('wind_kmh')

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

        row = {"Lega": fx['league'], "Data": date_iso, "Match": f"{home}-{away}"}
        row.update(pack("tiri", lam_h_s, vr_h_s, lam_a_s, vr_a_s, default_shots))
        row.update(pack("angoli", lam_h_c, vr_h_c, lam_a_c, vr_a_c, default_corners))
        if T is not None:
            row.update({"T°C": T, "Prec(mm)": P, "Vento(km/h)": W})
        rows.append(row)

    if rows:
        df_out = pd.DataFrame(rows)
        try:
            df_out['Data_s'] = pd.to_datetime(df_out['Data'], errors='coerce')
            df_out = df_out.sort_values(['Data_s', 'Lega', 'Match']).drop(columns=['Data_s'])
        except Exception:
            pass
        st.dataframe(df_out, use_container_width=True)
        st.download_button("Scarica CSV unico", df_out.to_csv(index=False).encode('utf-8'),
                           "dashboard_prob_calibrate.csv", "text/csv")

    if only_list:
        st.markdown("**Partite senza dati completi (mostrate comunque):**")
        st.write(pd.DataFrame(only_list)[['league','date','home','away']])

st.caption("Fonti: FBref (calendari), Open-Meteo (meteo), football-data.co.uk (storico). 'Forza aggiornamento dati' rigenera tutto subito.")
