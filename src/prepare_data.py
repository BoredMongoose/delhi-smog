"""Clean and combine the raw data into analysis-ready daily and hourly tables.

Inputs  (data/raw/):
  firms_viirs_punjab_haryana.csv   NASA VIIRS fire detections (fetch_data.py)
  weather_delhi_hourly.csv         Open-Meteo ERA5 hourly weather for Delhi (fetch_data.py)
  north_india_states.geojson       state boundaries (Natural Earth)
  kaggle/city_day.csv, city_hour.csv, station_day.csv, stations.csv   CPCB data via Kaggle

Outputs (data/processed/):
  fires_daily.csv     date, fire_count, fire_frp           - Punjab + Haryana only (burning season + winter)
  fires_grid.csv      year, lat, lon, fires                - 0.1° grid, for the map
  weather_daily.csv   date, wind_kmh, wind_u, wind_v, nw_wind_share, mixing_height_m, night_mixing_m,
                      temp_c, temp_range_c, rh_pct, rain_mm
  pm25_daily.csv      date, pm25_city, pm25_core10         - Delhi city mean; 10 long-running stations
  pm25_hourly.csv     datetime, pm25                       - Delhi city mean
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as Polygon

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/processed"
PROC.mkdir(parents=True, exist_ok=True)


def state_polygons(names):
    """Outer rings of each state's (multi)polygon as matplotlib Paths."""
    gj = json.loads((RAW / "north_india_states.geojson").read_text())
    rings = []
    for feat in gj["features"]:
        if feat["properties"]["name"] not in names:
            continue
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings += [Polygon(np.array(p[0])) for p in polys]
    return rings


def prepare_fires():
    f = pd.concat([pd.read_csv(RAW / "firms_viirs_punjab_haryana.csv"), pd.read_csv(RAW / "firms_viirs_winter.csv")])
    # nominal/high confidence, presumed vegetation fires (type 0); drop low-confidence detections
    f = f[f.confidence.isin(["n", "h"]) & (f.type == 0)].copy()
    pts = f[["longitude", "latitude"]].to_numpy()
    inside = np.zeros(len(f), dtype=bool)
    for ring in state_polygons({"Punjab", "Haryana"}):
        inside |= ring.contains_points(pts)
    print(f"fires: {len(f):,} in bounding box -> {inside.sum():,} inside Punjab + Haryana")
    f = f[inside]
    f["date"] = pd.to_datetime(f.acq_date)

    daily = (f.groupby("date").agg(fire_count=("frp", "size"), fire_frp=("frp", "sum")).reset_index())
    # make every day of each season explicit, so fire-free days are 0 rather than missing
    days = pd.concat([pd.DataFrame({"date": pd.date_range(f"{y}-09-15", f"{y}-12-15")}) for y in range(2015, 2026)]
                     + [pd.DataFrame({"date": pd.date_range(f"{y}-12-16", f"{y + 1}-03-31")}) for y in range(2015, 2020)])
    daily = days.merge(daily, on="date", how="left").fillna({"fire_count": 0, "fire_frp": 0})
    daily.to_csv(PROC / "fires_daily.csv", index=False)

    grid = (f.assign(year=f.date.dt.year, lat=(f.latitude // 0.1) * 0.1 + 0.05, lon=(f.longitude // 0.1) * 0.1 + 0.05)
             .groupby(["year", "lat", "lon"]).size().rename("fires").reset_index().round(3))
    grid.to_csv(PROC / "fires_grid.csv", index=False)
    print(daily.groupby(daily.date.dt.year).fire_count.sum().astype(int).to_string())


def prepare_weather():
    # smoke from Punjab/Haryana reaches Delhi on north-westerly winds (direction 270-360°)
    w = pd.read_csv(RAW / "weather_delhi_hourly.csv", parse_dates=["time"])
    w["nw"] = w.wind_direction_10m.between(270, 360)
    # wind as a vector (u = towards east, v = towards north) so directions average correctly
    rad = np.deg2rad(w.wind_direction_10m)
    w["u"], w["v"] = -w.wind_speed_10m * np.sin(rad), -w.wind_speed_10m * np.cos(rad)
    w["afternoon_mixing"] = w.boundary_layer_height.where(w.time.dt.hour.between(12, 16))
    w["night_mixing"] = w.boundary_layer_height.where(~w.time.dt.hour.between(7, 19))
    w["date"] = w.time.dt.normalize()
    weather = w.groupby("date").agg(
        wind_kmh=("wind_speed_10m", "mean"),
        wind_u=("u", "mean"),
        wind_v=("v", "mean"),
        nw_wind_share=("nw", "mean"),
        mixing_height_m=("afternoon_mixing", "mean"),   # how high pollution can mix during the day
        night_mixing_m=("night_mixing", "mean"),        # night-time inversion strength
        temp_c=("temperature_2m", "mean"),
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        rh_pct=("relative_humidity_2m", "mean"),
        rain_mm=("precipitation", "sum"),
    ).reset_index()
    weather["temp_range_c"] = weather.pop("temp_max") - weather.pop("temp_min")
    weather.to_csv(PROC / "weather_daily.csv", index=False)


def prepare_pm25():
    k = RAW / "kaggle"
    day = pd.read_csv(k / "city_day.csv", parse_dates=["Date"])
    day = day[day.City == "Delhi"][["Date", "PM2.5"]].rename(columns={"Date": "date", "PM2.5": "pm25_city"})

    # robustness series: the 10 Delhi stations that reported throughout 2015-2020, so the
    # average is not affected by new stations joining the network over time
    stations = pd.read_csv(k / "stations.csv")
    sd = pd.read_csv(k / "station_day.csv", parse_dates=["Date"])
    sd = sd[sd.StationId.isin(stations[stations.City == "Delhi"].StationId)].dropna(subset=["PM2.5"])
    first_seen = sd.groupby("StationId").Date.min()
    core = first_seen[first_seen < "2015-06-01"].index
    core10 = (sd[sd.StationId.isin(core)].groupby("Date")["PM2.5"].mean()
              .rename("pm25_core10").rename_axis("date").reset_index())
    print(f"core stations ({len(core)}):", ", ".join(stations.set_index("StationId").loc[core, "StationName"]
                                                  .str.replace(", Delhi.*", "", regex=True)))
    day.merge(core10, on="date", how="left").to_csv(PROC / "pm25_daily.csv", index=False)

    hour = pd.read_csv(k / "city_hour.csv", parse_dates=["Datetime"])
    hour = hour[hour.City == "Delhi"][["Datetime", "PM2.5"]].rename(columns={"Datetime": "datetime", "PM2.5": "pm25"})
    hour.to_csv(PROC / "pm25_hourly.csv", index=False)


if __name__ == "__main__":
    prepare_fires()
    prepare_weather()
    prepare_pm25()
