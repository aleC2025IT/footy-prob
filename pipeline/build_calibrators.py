import json, numpy as np, pandas as pd
from pathlib import Path
from pipeline.data_sources.football_data import load_league_data
from pipeline.compute_params import compute_team_metrics
from pipeline.modeling.prob_model import combine_strengths, blended_var_factor, finalize_probability
from pipeline.modeling.calibration import bin_and_fit

def run_build(settings_path='settings.yaml', out_path='app/public/data/calibrators.json'):
    import yaml
    with open(settings_path,'r') as f: cfg=yaml.safe_load(f)
    leagues=[l['code'] for l in cfg['leagues']]
    shots_grid=cfg['calibration_thresholds']['shots']
    corners_grid=cfg['calibration_thresholds']['corners']
    split_ratio=float(cfg.get('split_ratio',0.7))
    all_cals={}
    for code in leagues:
        df=load_league_data(code, cfg['seasons_back'])
        if df.empty: 
            continue
        df['Date']=pd.to_datetime(df['Date'], errors='coerce')
        df=df.dropna(subset=['Date']).sort_values('Date')
        n=len(df); cut=int(n*split_ratio)
        train=df.iloc[:cut].copy(); test=df.iloc[cut:].copy()

        metrics=compute_team_metrics(train, cfg['decay_half_life_games'])[code]
        league_means=metrics['league_means']; league_vr=metrics['league_var_ratio']; teams=metrics['teams']

        recs=[]
        def preds_for_game(home, away, yH_sh, yA_sh, yH_co, yA_co):
            if home not in teams or away not in teams: 
                return
            th, ta = teams[home], teams[away]

            # SHOTS
            lmean_s=(league_means['shots_home_for']+league_means['shots_away_for'])/2.0
            lamH_s=combine_strengths(th['shots_for_home'],league_means['shots_home_for'],ta['shots_against_away'],league_means['shots_away_against'],lmean_s,1.05)
            lamA_s=combine_strengths(ta['shots_for_away'],league_means['shots_away_for'],th['shots_against_home'],league_means['shots_home_against'],lmean_s,0.95)
            vrH_s=blended_var_factor(th.get('vr_shots_for_home'),ta.get('vr_shots_against_away'),league_vr.get('shots_home_for',1.1),1.0,2.0)
            vrA_s=blended_var_factor(ta.get('vr_shots_for_away'),th.get('vr_shots_against_home'),league_vr.get('shots_away_for',1.1),1.0,2.0)

            # CORNERS
            lmean_c=(league_means['corners_home_for']+league_means['corners_away_for'])/2.0
            lamH_c=combine_strengths(th['corners_for_home'],league_means['corners_home_for'],ta['corners_against_away'],league_means['corners_away_against'],lmean_c,1.03)
            lamA_c=combine_strengths(ta['corners_for_away'],league_means['corners_away_for'],th['corners_against_home'],league_means['corners_home_against'],lmean_c,0.97)
            vrH_c=blended_var_factor(th.get('vr_corners_for_home'),ta.get('vr_corners_against_away'),league_vr.get('corners_home_for',1.3),1.1,2.5)
            vrA_c=blended_var_factor(ta.get('vr_corners_for_away'),th.get('vr_corners_against_home'),league_vr.get('corners_away_for',1.3),1.1,2.5)

            # Per-squadra: shots & corners
            for k in shots_grid:
                pH=finalize_probability(int(k), lamH_s, var_factor=vrH_s, prefer_nb=True)
                pA=finalize_probability(int(k), lamA_s, var_factor=vrA_s, prefer_nb=True)
                recs.append(("shots",'home',k,pH, 1.0 if yH_sh>=k else 0.0))
                recs.append(("shots",'away',k,pA, 1.0 if yA_sh>=k else 0.0))
            for k in corners_grid:
                pH=finalize_probability(int(k), lamH_c, var_factor=vrH_c, prefer_nb=True)
                pA=finalize_probability(int(k), lamA_c, var_factor=vrA_c, prefer_nb=True)
                recs.append(("corners",'home',k,pH, 1.0 if yH_co>=k else 0.0))
                recs.append(("corners",'away',k,pA, 1.0 if yA_co>=k else 0.0))

            # Totale angoli (nuovo): somma di due conteggi
            lam_tot = lamH_c + lamA_c
            var_tot = lamH_c*vrH_c + lamA_c*vrA_c
            var_factor_tot = (var_tot/lam_tot) if lam_tot>0 else 1.0
            yTOT = (yH_co + yA_co)
            for k in corners_grid:
                pT = finalize_probability(int(k), lam_tot, var_factor=var_factor_tot, prefer_nb=True)
                recs.append(("corners_total",'total',k,pT, 1.0 if yTOT>=k else 0.0))

        for _, row in test.iterrows():
            try:
                preds_for_game(row['HomeTeam'], row['AwayTeam'],
                               int(row['HS']), int(row['AS']),
                               int(row['HC']), int(row['AC']))
            except Exception:
                continue

        if not recs:
            continue

        dfR=pd.DataFrame(recs, columns=['metric','side','k','p','y'])
        cal_map={}
        for metric in dfR['metric'].unique():
            cal_map.setdefault(metric,{})
            for side in dfR['side'].unique():
                cal_map[metric].setdefault(side,{})
                dms=dfR[(dfR['metric']==metric) & (dfR['side']==side)]
                for k in sorted(dms['k'].unique()):
                    d=dms[dms['k']==k]
                    xs,ys=bin_and_fit(d['p'].values, d['y'].values, bins=20)
                    cal_map[metric][side][str(int(k))]={'x':xs.tolist(),'y':ys.tolist()}
        all_cals[code]=cal_map

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(all_cals,f,indent=2)
    print("[OK] calibrators built")

if __name__=='__main__':
    run_build()
