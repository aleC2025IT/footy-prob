import requests, pandas as pd
from datetime import datetime, timedelta
SCHEDULE_URLS={
    "I1":"https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures",
    "E0":"https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "SP1":"https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures",
    "D1":"https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures"
}
def get_upcoming_fixtures(code, days=7):
    try:
        url=SCHEDULE_URLS[code]; res=requests.get(url,timeout=30); res.raise_for_status()
        dfs=pd.read_html(res.text); target=None
        for df in dfs:
            cols=[str(c).lower() for c in df.columns]
            if 'date' in cols and 'home' in cols and 'away' in cols: target=df; break
        if target is None: return pd.DataFrame(columns=['date','home','away'])
        df=target.copy()
        if 'Score' in df.columns: df=df[df['Score'].isna()]
        df['date']=pd.to_datetime(df['Date'],errors='coerce').dt.date
        df=df.dropna(subset=['date'])
        today=datetime.utcnow().date(); horizon=today+timedelta(days=max(1,days))
        df=df[(df['date']>=today) & (df['date']<=horizon)]
        out=df[['date','Home','Away']].rename(columns={'Home':'home','Away':'away'}); out['date']=out['date'].astype(str)
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['date','home','away'])