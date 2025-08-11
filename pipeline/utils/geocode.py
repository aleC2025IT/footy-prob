import requests, urllib.parse, json
from pathlib import Path
LEAGUE_COUNTRY = {"I1":"Italy","E0":"England","SP1":"Spain","D1":"Germany"}
STADIUMS_PATH = Path('app/public/data/stadiums.json')
def geocode_openmeteo(query, country_hint=None, count=1, language="en"):
    try:
        q=urllib.parse.quote(query)
        url=f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count={count}&language={language}&format=json"
        if country_hint: url += f"&country={urllib.parse.quote(country_hint)}"
        r=requests.get(url,timeout=15); r.raise_for_status(); js=r.json()
        results=js.get("results",[])
        if not results: return None
        best=results[0]
        return {"lat":best.get("latitude"),"lon":best.get("longitude")}
    except Exception:
        return None
def geocode_team_fallback(team_name, league_code, autosave=True):
    country=LEAGUE_COUNTRY.get(league_code)
    res=geocode_openmeteo(team_name, country_hint=country)
    if not res: res=geocode_openmeteo(f"{team_name} {country}" if country else team_name)
    if res and autosave:
        try:
            if STADIUMS_PATH.exists():
                with open(STADIUMS_PATH,'r',encoding='utf-8') as f: data=json.load(f)
            else:
                data={"I1":{},"E0":{},"SP1":{},"D1":{}}
            data.setdefault(league_code, {})[team_name]={'lat':res['lat'],'lon':res['lon']}
            STADIUMS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(STADIUMS_PATH,'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)
        except Exception: pass
    return res