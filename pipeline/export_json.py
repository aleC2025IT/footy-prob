# pipeline/export_json.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.data_sources.football_data import load_league_data

OUT = Path("app/public/data/league_team_metrics.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

LEAGUES = {"I1":"Serie A","E0":"Premier League","SP1":"La Liga","D1":"Bundesliga"}

def _safe_ratio(var, mean, floor=0.9, ceil=3.0):
    if mean is None or mean <= 0: return 1.0
    if var is None or var <= 0: return 1.0
    x = float(var) / float(mean)
    return float(max(floor, min(ceil, x)))

def _team_stats(df: pd.DataFrame, team: str):
    # home side (team = home): for = HS, against = AS
    d_home = df[df["home"] == team]
    d_away = df[df["away"] == team]
    # medie
    shots_for_home     = d_home["HS"].mean()
    shots_against_home = d_home["AS"].mean()
    shots_for_away     = d_away["AS"].mean()
    shots_against_away = d_away["HS"].mean()
    corners_for_home     = d_home["HC"].mean()
    corners_against_home = d_home["AC"].mean()
    corners_for_away     = d_away["AC"].mean()
    corners_against_away = d_away["HC"].mean()
    # var/mean (overdispersion) – robusto
    vr_shots_for_home       = _safe_ratio(d_home["HS"].var(ddof=1), d_home["HS"].mean(), 1.0, 2.0)
    vr_shots_against_home   = _safe_ratio(d_home["AS"].var(ddof=1), d_home["AS"].mean(), 1.0, 2.0)
    vr_shots_for_away       = _safe_ratio(d_away["AS"].var(ddof=1), d_away["AS"].mean(), 1.0, 2.0)
    vr_shots_against_away   = _safe_ratio(d_away["HS"].var(ddof=1), d_away["HS"].mean(), 1.0, 2.0)
    vr_corners_for_home     = _safe_ratio(d_home["HC"].var(ddof=1), d_home["HC"].mean(), 1.1, 2.5)
    vr_corners_against_home = _safe_ratio(d_home["AC"].var(ddof=1), d_home["AC"].mean(), 1.1, 2.5)
    vr_corners_for_away     = _safe_ratio(d_away["AC"].var(ddof=1), d_away["AC"].mean(), 1.1, 2.5)
    vr_corners_against_away = _safe_ratio(d_away["HC"].var(ddof=1), d_away["HC"].mean(), 1.1, 2.5)

    return {
        "shots_for_home": shots_for_home, "shots_against_home": shots_against_home,
        "shots_for_away": shots_for_away, "shots_against_away": shots_against_away,
        "corners_for_home": corners_for_home, "corners_against_home": corners_against_home,
        "corners_for_away": corners_for_away, "corners_against_away": corners_against_away,
        "vr_shots_for_home": vr_shots_for_home, "vr_shots_against_home": vr_shots_against_home,
        "vr_shots_for_away": vr_shots_for_away, "vr_shots_against_away": vr_shots_against_away,
        "vr_corners_for_home": vr_corners_for_home, "vr_corners_against_home": vr_corners_against_home,
        "vr_corners_for_away": vr_corners_for_away, "vr_corners_against_away": vr_corners_against_away,
    }

def build_one_league(code: str):
    df = load_league_data(code, include_current=True, only_last_season=True)
    if df.empty:
        return None
    # medie di lega
    league_means = {
        "shots_home_for":   df["HS"].mean(),
        "shots_away_for":   df["AS"].mean(),
        "shots_home_against": df["AS"].mean(),  # simmetrico
        "shots_away_against": df["HS"].mean(),
        "corners_home_for": df["HC"].mean(),
        "corners_away_for": df["AC"].mean(),
        "corners_home_against": df["AC"].mean(),
        "corners_away_against": df["HC"].mean(),
    }
    league_vr = {
        "shots_home_for":   _safe_ratio(df["HS"].var(ddof=1), df["HS"].mean(), 1.0, 2.0),
        "shots_away_for":   _safe_ratio(df["AS"].var(ddof=1), df["AS"].mean(), 1.0, 2.0),
        "corners_home_for": _safe_ratio(df["HC"].var(ddof=1), df["HC"].mean(), 1.1, 2.5),
        "corners_away_for": _safe_ratio(df["AC"].var(ddof=1), df["AC"].mean(), 1.1, 2.5),
    }
    teams = sorted(set(df["home"]).union(set(df["away"])))
    teams_dict = {t: _team_stats(df, t) for t in teams}
    return {"league_means": league_means, "league_var_ratio": league_vr, "teams": teams_dict}

def main():
    out = {}
    for code in LEAGUES.keys():
        info = build_one_league(code)
        if info is not None:
            out[code] = info
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} with {len(out)} leagues.")

if __name__ == "__main__":
    main()
