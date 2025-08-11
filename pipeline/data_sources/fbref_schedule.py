import re, requests, pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE_URLS = {
    "I1":  "https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures",
    "E0":  "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "SP1": "https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures",
    "D1":  "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures"
}

def _season_slug_today():
    # stagione europea: agosto → anno-anno+1
    today = datetime.now(ZoneInfo("Europe/Rome")).date()
    y = today.year
    # da luglio in poi parte la nuova
    start = 7
    start_year = y if today.month >= start else (y - 1)
    return f"{start_year}-{start_year+1}"

def _pick_target_url(code: str) -> str:
    """Se la pagina base è vecchia, cerca il link della stagione corrente (es. 2025-2026) e usalo."""
    url = BASE_URLS[code]
    try:
        html = requests.get(url, timeout=25).text
    except Exception:
        return url
    season = _season_slug_today()
    # cerca un link 'schedule/...Scores-and-Fixtures' che contenga la stagione corrente
    m = re.search(r'href="(/en/comps/\d+/\d{4}-\d{4}/schedule/[^"]*Scores-and-Fixtures[^"]*)"', html)
    if m and season in m.group(1):
        return "https://fbref.com" + m.group(1)
    return url  # fallback: base

def get_upcoming_fixtures(code, days=7):
    try:
        code = str(code).upper().strip()
        url = _pick_target_url(code)
        res = requests.get(url, timeout=30)
        res.raise_for_status()

        # prendi la tabella principale con Date / Home / Away
        dfs = pd.read_html(res.text)
        target = None
        for df in dfs:
            cols = [str(c).strip().lower() for c in df.columns]
            if 'date' in cols and any('home' in c for c in cols) and any('away' in c for c in cols):
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

        # parse date e filtro orizzonte in Europe/Rome
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        df = df.dropna(subset=['date'])
        today = datetime.now(ZoneInfo("Europe/Rome")).date()
        horizon = today + timedelta(days=max(1, int(days)))
        df = df[(df['date'] >= today) & (df['date'] <= horizon)]

        # pulizia stringhe
        df['home'] = df['home'].astype(str).str.strip()
        df['away'] = df['away'].astype(str).str.strip()

        return df[['date','home','away']].drop_duplicates().reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=['date','home','away'])
