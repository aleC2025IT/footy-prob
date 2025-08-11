import requests, pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SCHEDULE_URLS = {
    "I1":  "https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures",
    "E0":  "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "SP1": "https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures",
    "D1":  "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures"
}

def get_upcoming_fixtures(code, days=7):
    try:
        url = SCHEDULE_URLS[code]
        res = requests.get(url, timeout=30)
        res.raise_for_status()
        dfs = pd.read_html(res.text)
        target = None
        for df in dfs:
            cols = [str(c).strip().lower() for c in df.columns]
            if 'date' in cols and ('home' in cols or 'home team' in cols) and ('away' in cols or 'away team' in cols):
                target = df
                break
        if target is None:
            return pd.DataFrame(columns=['date','home','away'])

        df = target.copy()
        # normalizza nomi colonne
        ren = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc == 'date': ren[c] = 'date'
            elif 'home' in lc: ren[c] = 'home'
            elif 'away' in lc: ren[c] = 'away'
        df = df.rename(columns=ren)
        keep = [c for c in ['date','home','away'] if c in df.columns]
        df = df[keep].dropna(subset=['date','home','away'])

        # parse date e filtro orizzonte in fuso Europe/Rome
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        today = datetime.now(ZoneInfo("Europe/Rome")).date()
        horizon = today + timedelta(days=max(1, int(days)))
        df = df[(df['date'] >= today) & (df['date'] <= horizon)]

        # pulizia stringhe
        df['home'] = df['home'].astype(str).str.strip()
        df['away'] = df['away'].astype(str).str.strip()

        return df[['date','home','away']].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['date','home','away'])
