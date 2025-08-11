# pipeline/data_sources/football_data.py
import pandas as pd
import requests
from io import StringIO
from datetime import date

# Mappe codici lega uguali a football-data.co.uk
LEAGUE_FILES = {"E0": "E0", "SP1": "SP1", "D1": "D1", "I1": "I1"}

def _season_pair_last(today=None):
    """Ritorna (aa, bb) due cifre per la STAGIONE PASSATA (es. 23,24)."""
    if today is None: today = date.today()
    start_year = today.year if today.month >= 7 else (today.year - 1)  # stagione corrente
    last_start = start_year - 1
    return last_start % 100, (last_start + 1) % 100

def _season_pair_current(today=None):
    """Ritorna (aa, bb) per la STAGIONE CORRENTE (es. 24,25)."""
    if today is None: today = date.today()
    start_year = today.year if today.month >= 7 else (today.year - 1)
    return start_year % 100, (start_year + 1) % 100

def _fetch_fd_csv(league_code: str, pair: tuple[int, int]) -> pd.DataFrame:
    """Scarica un CSV football-data della stagione indicata e normalizza le colonne base."""
    lf = LEAGUE_FILES.get(league_code.upper())
    if not lf:
        return pd.DataFrame()
    y1, y2 = pair
    url = f"https://www.football-data.co.uk/mmz4281/{y1:02d}{y2:02d}/{lf}.csv"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.read_csv(StringIO(r.text))
    # colonne standard più usate (variano poco nelle ultime stagioni)
    needed = ["Date", "HomeTeam", "AwayTeam", "HS", "AS", "HC", "AC"]
    for col in needed:
        if col not in df.columns:
            # se manca qualcosa, rinuncio per non rompere la pipeline
            return pd.DataFrame()
    # parse date (formato dd/mm/yy)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    # rinomina in modo uniforme
    df = df.rename(columns={
        "Date": "date", "HomeTeam": "home", "AwayTeam": "away",
        "HS": "HS", "AS": "AS", "HC": "HC", "AC": "AC"
    })
    return df[["date", "home", "away", "HS", "AS", "HC", "AC"]].copy()

def load_league_data(league_code: str,
                     include_current: bool = True,
                     only_last_season: bool = True) -> pd.DataFrame:
    """
    Ritorna SOLO la stagione passata (+ la corrente se include_current=True).
    Colonne: date, home, away, HS (tiri casa), AS (tiri trasferta), HC/AC (angoli).
    """
    df_list = []
    # stagione passata (obbligatoria)
    df_last = _fetch_fd_csv(league_code, _season_pair_last())
    if not df_last.empty:
        df_list.append(df_last)
    # stagione corrente (opzionale, per aggiornarsi via via)
    if include_current:
        df_curr = _fetch_fd_csv(league_code, _season_pair_current())
        if not df_curr.empty:
            df_list.append(df_curr)
    if not df_list:
        return pd.DataFrame(columns=["date","home","away","HS","AS","HC","AC"])
    df = pd.concat(df_list, ignore_index=True)
    # ordina cronologicamente
    df = df.sort_values("date").reset_index(drop=True)
    return df
