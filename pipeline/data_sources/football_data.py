import io, requests, pandas as pd
from datetime import datetime
BASE="https://www.football-data.co.uk/mmz4281"
LEAGUE_FILES={'E0':'E0.csv','I1':'I1.csv','D1':'D1.csv','SP1':'SP1.csv'}
def season_code(y): return f"{str(y)[-2:]}{str((y+1))[-2:]}"
def fetch_csv(code, y):
    sc=season_code(y); url=f"{BASE}/{sc}/{LEAGUE_FILES[code]}"
    r=requests.get(url,timeout=30); r.raise_for_status()
    df=pd.read_csv(io.BytesIO(r.content)); df['League']=code; df['SeasonStart']=y; return df
def load_league_data(code, seasons_back=2):
    now=datetime.utcnow(); season_start= now.year if now.month>=7 else now.year-1
    frames=[]
    for k in range(seasons_back,-1,-1):
        yr=season_start-k
        try: frames.append(fetch_csv(code, yr))
        except Exception as e: print(f"[WARN] {code} {yr}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()