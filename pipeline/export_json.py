import os, json, pandas as pd
from pathlib import Path
from pipeline.data_sources.football_data import load_league_data
from pipeline.compute_params import compute_team_metrics, save_json
def run_export(settings_path='settings.yaml', out_dir='app/public/data'):
    import yaml
    with open(settings_path,'r') as f: cfg=yaml.safe_load(f)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    all_data={}
    for lg in cfg['leagues']:
        code=lg['code']
        df=load_league_data(code, cfg['seasons_back'])
        if df.empty: print(f"[WARN] No data for {code}"); continue
        metrics=compute_team_metrics(df, cfg['decay_half_life_games'])
        all_data[code]=metrics[code]
    save_json(all_data, os.path.join(out_dir,'league_team_metrics.json'))
    print("[OK] metrics exported")
if __name__=='__main__': run_export()