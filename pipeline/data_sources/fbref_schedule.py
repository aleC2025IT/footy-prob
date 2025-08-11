import re, time, requests, pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from functools import lru_cache

BASE_URLS = {
    "I1":  "https://fbref.com/en/comps/11/schedule/Serie-A-Scores-and-Fixtures",
    "E0":  "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures",
    "SP1": "https://fbref.com/en/comps/12/schedule/La-Liga-Scores-and-Fixtures",
    "D1":  "https://fbref.com/en/comps/20/schedule/Bundesliga-Scores-and-Fixtures",
}

def _season_slug_today():
    today = datetime.now(ZoneInfo("Europe/Rome")).date()
    start_year = today.year if today.month >= 7 else (today.year - 1)
    return f"{start_year}-{start_year+1}"

def _session():
    s = requests.Session()
    s.headers.update({
        # User-Agent “normale” per evitare blocchi stupidi
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36"
    })
    return s

def _pick_current_season_url(code: str, sess: requests.Session) -> str:
    base = BASE_URLS[code]
    try:
        html = sess.get(base, timeout=25).text
        season = _season_slug_today()
        m = re.search(r'href="(/en/comps/\d+/\d{4}-\d{4}/schedule/[^"]*Scores-and-Fixtures[^"]*)"', html)
        if m and season in m.group(1):
            return "https://fbref.com" + m.group(1)
        return base
    except Exception:
        return base

def _get_html_with_retry(sess: requests.Session, url: str, max_retry: int = 2) -> str:
    last_err = ""
    for i in range(max_retry + 1):
        try:
            r = sess.get(url, timeout=30)
            if r.status_code == 429:
                # rate limit: aspetta un attimo e riprova
                wait = int(r.headers.get("Retry-After", "3"))
                time.sleep(min(wait, 5))
                last_err = "HTTP 429"
                continue
            r.raise_for_status()
            return r.text
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            last_err = f"HTTP {code}" if code else "HTTP error"
            if code in (429, 503) and i < max_retry:
                time.sleep(3)
                continue
            break
        except Exception:
            last_err = "conn error"
            if i < max_retry:
                time.sleep(2)
                continue
            break
    raise RuntimeError(last_err)

@lru_cache(maxsize=32)
def get_upcoming_fixtures(code: str, days: int = 7) -> pd.DataFrame:
    """Restituisce DataFrame con colonne: date (yyyy-mm-dd), home, away."""
    try:
        code = str(code).upper().strip()
        sess = _session()
        url = _pick_current_season_url(code, sess)
        html = _get_html_with_retry(sess, url)
        # prendi tabella che contiene Date/Home/Away
        dfs = pd.read_html(html)
        target = None
        for df in dfs:
            cols = [str(c).strip().lower() for c in df.columns]
            if 'date' in cols and any('home' in c for c in cols) and any('away' in c for c in cols):
                target = df; break
        if target is None:
            return pd.DataFrame(columns=['date','home','away'])

        df = target.copy()
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

        df['home'] = df['home'].astype(str).str.strip()
        df['away'] = df['away'].astype(str).str.strip()
        return df[['date','home','away']].drop_duplicates().reset_index(drop=True)
    except Exception:
        # fallisci “morbido” con DF vuoto (lo gestisce l’app mostrando la partita come lista)
        return pd.DataFrame(columns=['date','home','away'])
