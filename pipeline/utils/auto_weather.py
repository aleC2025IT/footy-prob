import json, requests
from pathlib import Path
LEAGUE_TZ = {"I1":"Europe/Rome","E0":"Europe/London","SP1":"Europe/Madrid","D1":"Europe/Berlin"}
CACHE_PATH = Path('app/public/data/cache_meteo.json')
def _load_cache():
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH,'r',encoding='utf-8') as f: return json.load(f)
        except Exception: return {}
    return {}
def _save_cache(obj):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH,'w',encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)
def fetch_openmeteo_conditions(lat, lon, date_iso, hour_local=18, tz="Europe/Rome"):
    key = f"{round(lat,4)}|{round(lon,4)}|{date_iso}|{hour_local}|{tz}"
    cache=_load_cache()
    if key in cache: return cache[key]
    try:
        url=("https://api.open-meteo.com/v1/forecast"
             f"?latitude={lat}&longitude={lon}"
             f"&hourly=temperature_2m,precipitation,wind_speed_10m,snowfall"
             f"&start_date={date_iso}&end_date={date_iso}&timezone={tz}")
        r=requests.get(url,timeout=20); r.raise_for_status(); js=r.json()
        hours=js.get('hourly',{}).get('time',[])
        temps=js.get('hourly',{}).get('temperature_2m',[])
        precs=js.get('hourly',{}).get('precipitation',[])
        winds=js.get('hourly',{}).get('wind_speed_10m',[])
        snows=js.get('hourly',{}).get('snowfall',[])
        if not hours: return {}
        idx=min(range(len(hours)), key=lambda i: abs(int(hours[i].split('T')[1].split(':')[0]) - hour_local))
        temp=temps[idx] if idx<len(temps) else None
        prcp=precs[idx] if idx<len(precs) else None
        wind=winds[idx] if idx<len(winds) else None
        snow=snows[idx] if idx<len(snows) else None
        rain=(prcp or 0)>=0.2; snowf=(snow or 0)>0; wind_strong=(wind or 0)>=25
        hot=(temp or 0)>=28; cold=(temp or 0)<=3
        out={'rain':rain,'snow':snowf,'wind_strong':wind_strong,'hot':hot,'cold':cold,
             'meta':{'temp_c':temp,'precip_mm':prcp,'wind_kmh':wind,'snow_mm':snow}}
        cache[key]=out; _save_cache(cache); return out
    except Exception:
        return {}