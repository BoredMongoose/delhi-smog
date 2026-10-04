"""Download the open data this project needs (files that already exist are skipped).

1. NASA FIRMS VIIRS (Suomi-NPP) fire detections around Punjab & Haryana
   - burning season: 15 Sep - 15 Dec, 2015-2025
   - winter control period: 16 Dec - 31 Mar, 2015-16 to 2019-20 (to confirm it is fire-free)
2. Open-Meteo ERA5 hourly weather for central Delhi, 2015-2025

The FIRMS key is read from .env (FIRMS_MAP_KEY=...), which is git-ignored.
"""
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

KEY = dict(line.split("=", 1) for line in (ROOT / ".env").read_text().split())["FIRMS_MAP_KEY"]
# west, south, east, north - covers Punjab and Haryana (trimmed to state borders in prepare_data.py)
BBOX = "73.8,27.6,77.6,32.6"


def fetch_fires(windows, out):
    if out.exists():
        print("exists, skipping:", out.name)
        return
    frames = []
    for start, end in windows:
        day = start
        while day <= end:
            span = min(5, (end - day).days + 1)  # the API returns at most 5 days per call
            for source in ("VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"):  # standard archive, then near-real-time
                url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{KEY}/{source}/{BBOX}/{span}/{day}"
                text = requests.get(url, timeout=120).text
                if text.startswith("latitude") and text.count("\n") > 1:
                    frames.append(pd.read_csv(io.StringIO(text)).assign(source=source))
                    break
            day += timedelta(days=span)
            time.sleep(0.5)
        print(start.year, "fires so far:", sum(len(f) for f in frames))
    pd.concat(frames).drop_duplicates().to_csv(out, index=False)


def fetch_weather(out):
    if out.exists():
        print("exists, skipping:", out.name)
        return
    frames = []
    for year in range(2015, 2026):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=120, params=dict(
            latitude=28.61, longitude=77.21, start_date=f"{year}-01-01", end_date=f"{year}-12-31",
            hourly="wind_speed_10m,wind_direction_10m,boundary_layer_height,temperature_2m,"
                   "relative_humidity_2m,precipitation",
            timezone="Asia/Kolkata")).json()
        frames.append(pd.DataFrame(r["hourly"]))
        time.sleep(1)
    pd.concat(frames).to_csv(out, index=False)
    print("weather rows:", sum(len(f) for f in frames))


if __name__ == "__main__":
    fetch_weather(RAW / "weather_delhi_hourly.csv")
    fetch_fires([(date(y, 9, 15), date(y, 12, 15)) for y in range(2015, 2026)],
                RAW / "firms_viirs_punjab_haryana.csv")
    fetch_fires([(date(y, 12, 16), date(y + 1, 3, 31)) for y in range(2015, 2020)],
                RAW / "firms_viirs_winter.csv")
