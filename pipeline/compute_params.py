import json, pandas as pd
from pathlib import Path
from pipeline.utils.decay import weighted_mean_var
def _wm(s,hl): m,_=weighted_mean_var(pd.to_numeric(s,errors='coerce').to_numpy(), hl); return m
def _vr(s,hl):
    import numpy as np
    x=pd.to_numeric(s,errors='coerce').dropna().to_numpy()
    if len(x)==0: return float('nan')
    m,v=weighted_mean_var(x,hl)
    if m is None or v is None or m<=0: return float('nan')
    return float(v/m)
def compute_team_metrics(df, decay_half_life_games=12):
    need=['Date','HomeTeam','AwayTeam','HS','AS','HC','AC','League','SeasonStart']
    for c in need:
        if c not in df.columns: raise ValueError(f"Missing column {c}")
    df=df.copy(); df['Date']=pd.to_datetime(df['Date'],errors='coerce')
    df=df.dropna(subset=['Date']).sort_values('Date'); out={}
    for lg in df['League'].unique():
        d=df[df['League']==lg].copy()
        league_means={'shots_home_for':d['HS'].mean(),'shots_away_for':d['AS'].mean(),'shots_home_against':d['AS'].mean(),'shots_away_against':d['HS'].mean(),
                      'corners_home_for':d['HC'].mean(),'corners_away_for':d['AC'].mean(),'corners_home_against':d['AC'].mean(),'corners_away_against':d['HC'].mean()}
        def _vrL(col):
            s=pd.to_numeric(d[col],errors='coerce').dropna()
            if s.empty: return float('nan')
            m=s.mean(); v=s.var(ddof=1) if len(s)>1 else 0.0
            return float(v/m) if m>0 else float('nan')
        league_var_ratio={'shots_home_for':_vrL('HS'),'shots_away_for':_vrL('AS'),'corners_home_for':_vrL('HC'),'corners_away_for':_vrL('AC')}
        teams=pd.unique(pd.concat([d['HomeTeam'],d['AwayTeam']]))
        stats={}
        for t in teams:
            h=d[d['HomeTeam']==t]; a=d[d['AwayTeam']==t]
            stats[t]={'shots_for_home':_wm(h['HS'],12),'shots_for_away':_wm(a['AS'],12),'shots_against_home':_wm(h['AS'],12),'shots_against_away':_wm(a['HS'],12),
                      'corners_for_home':_wm(h['HC'],12),'corners_for_away':_wm(a['AC'],12),'corners_against_home':_wm(h['AC'],12),'corners_against_away':_wm(a['HC'],12),
                      'vr_shots_for_home':_vr(h['HS'],12),'vr_shots_for_away':_vr(a['AS'],12),'vr_shots_against_home':_vr(h['AS'],12),'vr_shots_against_away':_vr(a['HS'],12),
                      'vr_corners_for_home':_vr(h['HC'],12),'vr_corners_for_away':_vr(a['AC'],12),'vr_corners_against_home':_vr(h['AC'],12),'vr_corners_against_away':_vr(a['HC'],12),
                      'n_games':int(len(h)+len(a))}
        out[lg]={'league_means':league_means,'league_var_ratio':league_var_ratio,'teams':stats}
    return out
def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)