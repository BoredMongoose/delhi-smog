# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Who's really choking Delhi: Diwali crackers or Punjab's stubble fires?
#
# Every November, Delhi's air becomes some of the worst on Earth, and every November the same argument
# starts: is it the **firecrackers** of Diwali, or the **crop-residue fires** in Punjab and Haryana?
#
# This notebook tests both with five seasons of measured air-quality data (2015–2019), over 600,000 NASA
# satellite fire detections and hourly weather reanalysis.
#
# **Short answer:**
# 1. **Diwali night** is one of the five most polluted nights of every year (hourly PM2.5 of 500–800 µg/m³),
#    but the spike is short. Over the season it adds roughly 2% as much pollution as the stubble-burning period.
# 2. **Stubble-season smoke** is the main driver in the first half of November: about **60%** of the
#    PM2.5 Delhi breathes from 1–10 November, and about **40%** from mid-October to the end of November.
# 3. **Over the whole winter**, Delhi's own pollution trapped by winter weather is the biggest share,
#    at roughly **80–90%**. That's why December and January stay toxic long after the fires stop.
#
# Data sources: CPCB station data (via Kaggle), NASA FIRMS VIIRS fire detections and Open-Meteo ERA5 weather.
# Run `python src/fetch_data.py && python src/prepare_data.py` first.

# %%
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import patheffects
from matplotlib.colors import LogNorm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from style import BLUE, BLUE_LIGHT, GRID, INK, INK_2, NEUTRAL, ORANGE, ORANGE_LIGHT, VIOLET, footnote, save, titles  # noqa: E402

PROC = ROOT / "data" / "processed"
pd.set_option("display.precision", 1)

pm = pd.read_csv(PROC / "pm25_daily.csv", parse_dates=["date"])
weather = pd.read_csv(PROC / "weather_daily.csv", parse_dates=["date"])
fires = pd.read_csv(PROC / "fires_daily.csv", parse_dates=["date"])
hourly = pd.read_csv(PROC / "pm25_hourly.csv", parse_dates=["datetime"])

daily = pm.merge(weather, on="date").merge(fires, on="date", how="left")
daily["season"] = np.where(daily.date.dt.month >= 7, daily.date.dt.year, daily.date.dt.year - 1)  # Oct 2018-Mar 2019 = 2018
daily["md"] = daily.date.dt.month * 100 + daily.date.dt.day          # month-day, e.g. 1105 = 5 Nov
daily["dow"] = daily.date.dt.dayofweek
daily["lmix"] = np.log(daily.mixing_height_m)                          # afternoon mixing height (log)
daily["lnight"] = np.log(daily.night_mixing_m)                         # night-time mixing height (log)

DIWALI = pd.to_datetime(["2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27"])
SEASONS = range(2015, 2020)
print(f"{len(daily):,} days of PM2.5 + weather, {daily.date.min():%b %Y} – {daily.date.max():%b %Y}")
print(f"{fires.fire_count.sum():,.0f} fire detections in Punjab + Haryana, 2015–2025")

# %% [markdown]
# ## 1. The fires
#
# NASA's VIIRS satellite instrument flags every hot spot it sees in Punjab and Haryana. Burning peaks
# from late October to mid-November, between the rice harvest and wheat sowing.

# %%
season_fires = (fires[fires.date.dt.month.isin([9, 10, 11, 12]) & (fires.date.dt.month * 100 + fires.date.dt.day).between(915, 1215)]
                .groupby(fires.date.dt.year).fire_count.sum())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(season_fires.index, season_fires.values / 1000, color=ORANGE, width=0.65, edgecolor="#fcfcfb", linewidth=2)
ax.bar_label(bars, labels=[f"{v / 1000:.0f}k" for v in season_fires.values], padding=3, fontsize=9.5)
ax.set_xticks(season_fires.index)
ax.set_ylabel("Fire detections (thousands)")
ax.grid(axis="x", visible=False)
drop = 1 - season_fires[2025] / season_fires[2021]
titles(ax, f"Satellite-detected crop fires fell {drop:.0%} from 2021 to 2025",
       "Fire detections in Punjab and Haryana, 15 Sep – 15 Dec. Part of the drop may be farmers burning after the satellite's afternoon pass.")
footnote(fig, "Data: NASA FIRMS, VIIRS (Suomi-NPP), nominal and high confidence vegetation fires.", y=-0.03)
save(fig, "01_fires_by_year.png")
plt.show()

# %%
gj = json.loads((ROOT / "data/raw/north_india_states.geojson").read_text())
grid = pd.read_csv(PROC / "fires_grid.csv")
grid = grid[grid.year.between(2015, 2019)].groupby(["lat", "lon"]).fires.sum().reset_index()
lons, lats = np.round(np.arange(73.8, 77.65, 0.1), 2), np.round(np.arange(27.6, 32.65, 0.1), 2)
Z = np.zeros((len(lats) - 1, len(lons) - 1))
for _, r in grid.iterrows():
    i, j = int((r.lat - 27.6) // 0.1), int((r.lon - 73.8) // 0.1)
    if 0 <= i < Z.shape[0] and 0 <= j < Z.shape[1]:
        Z[i, j] += r.fires

fig, ax = plt.subplots(figsize=(8.5, 9.5))
mesh = ax.pcolormesh(lons, lats, np.ma.masked_equal(Z, 0), cmap="Oranges", norm=LogNorm(vmin=10, vmax=Z.max()))
for feat in gj["features"]:
    geom = feat["geometry"]
    for poly in (geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]):
        ring = np.array(poly[0])
        ax.plot(ring[:, 0], ring[:, 1], color=INK_2, lw=0.8)
halo = [patheffects.withStroke(linewidth=3.5, foreground="white")]   # keeps labels readable over the fire cells
for name, (x, y) in {"PUNJAB": (74.75, 31.45), "HARYANA": (76.05, 29.25), "RAJASTHAN": (74.4, 28.1),
                     "HIMACHAL\nPRADESH": (76.9, 32.0), "UTTAR\nPRADESH": (77.45, 28.0)}.items():
    ax.text(x, y, name, color=INK_2, fontsize=9.5, ha="center", fontweight="bold", path_effects=halo, zorder=6)
ax.plot(77.21, 28.61, marker="*", ms=16, color=INK, zorder=5)
ax.text(77.21, 28.43, "Delhi", ha="center", fontsize=11, fontweight="bold", path_effects=halo, zorder=6)
ax.annotate("", xy=(77.05, 28.78), xytext=(75.6, 30.6),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2, mutation_scale=18, path_effects=halo), zorder=6)
ax.text(75.0, 29.05, "North-westerly winds\ncarry the smoke\n~300 km to Delhi", fontsize=10, ha="center", va="center",
        path_effects=halo, zorder=6)
ax.set_xlim(73.8, 77.6); ax.set_ylim(27.6, 32.6)
ax.set_aspect(1 / np.cos(np.deg2rad(30)))
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)
cb = fig.colorbar(mesh, ax=ax, shrink=0.45, pad=0.02)
cb.set_label("Fire detections per 0.1° cell, 2015–2019", color=INK_2)
cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
cb.outline.set_visible(False)
titles(ax, "Where the fires are", "Every satellite fire detection in Punjab and Haryana, 2015–2019 burning seasons")
footnote(fig, "Data: NASA FIRMS VIIRS; boundaries: Natural Earth.", y=0.06)
save(fig, "02_fire_map.png")
plt.show()

# %% [markdown]
# ## 2. Why a simple correlation misleads
#
# The obvious approach is to regress Delhi's daily PM2.5 on the number of fires. That gives a large, highly
# "significant" fire effect. The **placebo test** below gives the model the *wrong year's* fires (same
# calendar dates, previous or next year). Fake fires "explain" pollution just as well as real ones.
#
# The reason is that fires and smog both peak in November, partly because November weather traps
# pollution. Daily fire counts can't separate the smoke from the season, so a different method is needed.

# %%
fire_by_date = fires.set_index("date").fire_count


def fires_around(dates, year_shift=0):
    """Average fires on day t and t-1 (smoke takes ~a day to arrive), optionally from another year."""
    get = lambda d: fire_by_date.get(d - pd.DateOffset(years=year_shift), np.nan)
    return dates.map(lambda d: (get(d) + get(d - pd.Timedelta(days=1))) / 2 / 1000)


autumn = daily[(daily.md >= 1001) & (daily.md <= 1130) & daily.season.isin(SEASONS)].copy()
autumn["diwali"] = autumn.date.isin(DIWALI).astype(int)
autumn["diwali_next"] = autumn.date.isin(DIWALI + pd.Timedelta(days=1)).astype(int)
VARIANTS = {"f_real": ("real fires", 0), "f_prev": ("placebo: previous year's fires", 1), "f_next": ("placebo: next year's fires", -1)}
for col, (_, shift) in VARIANTS.items():
    autumn[col] = fires_around(autumn.date, shift)
sample = autumn.dropna(subset=list(VARIANTS))

rows = []
for col, (label, _) in VARIANTS.items():
    m = smf.ols(f"pm25_city ~ {col} + diwali + diwali_next + lmix + wind_kmh + rain_mm + temp_c + C(season)",
                data=sample).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    rows.append({"fire variable": label, "µg/m³ per 1,000 fires/day": m.params[col],
                 "95% CI low": m.conf_int().loc[col, 0], "95% CI high": m.conf_int().loc[col, 1], "R²": m.rsquared})
pd.DataFrame(rows).set_index("fire variable")

# %% [markdown]
# ## 3. A better method: weather normalisation
#
# Instead of asking *"do more fires mean more smog?"*, the question becomes *"how much worse is the air
# than the weather alone would predict?"*
#
# 1. **Train a weather model on fire-free months.** From 16 December to 31 March there's almost no burning
#    (about 1–2% of autumn's fire count), but Delhi's own emissions continue and winter weather traps them.
#    A model fitted there learns how weather affects Delhi's own pollution.
# 2. **Predict October–November from the weather alone.** This is what the air would look like with
#    Delhi's usual emissions under that season's weather.
# 3. **The gap between actual and predicted is the autumn excess.** That's pollution only autumn has:
#    stubble smoke, plus Diwali.
#
# The model is a log-linear regression on daily weather: afternoon and night mixing height (how high
# pollution can disperse), wind speed and direction, temperature, diurnal range, humidity, rain, day of
# week, and a separate level for each season. A gradient-boosting model was also tested. It did worse
# out of sample (R² 0.25 against 0.43 when whole winters are held out), because ~600 training days aren't
# enough for it.

# %%
WEATHER_TERMS = "wind_kmh + wind_u + wind_v + lmix + lnight + temp_c + temp_range_c + rh_pct + rain_mm + C(dow)"
covid = daily.date >= "2020-03-22"   # lockdown changed emissions: keep it out of training


def weather_model(y="pm25_city", train_mask=None, formula=None):
    """Fit log(PM2.5) ~ weather on fire-free months; return (model, smearing factor)."""
    if train_mask is None:
        train_mask = (daily.md >= 1216) | (daily.md <= 331)
    train = daily[train_mask & ~covid & daily[y].notna()]
    model = smf.ols(formula or f"np.log({y}) ~ {WEATHER_TERMS} + C(season)", data=train).fit()
    return model, np.mean(np.exp(model.resid)), train   # smearing corrects the bias of exp(mean of log)


model, smear, train = weather_model()

# cross-validation: hold out one month of one winter at a time
cv_pred = pd.Series(index=train.index, dtype=float)
for _, block in train.groupby(["season", train.date.dt.month]):
    m = smf.ols(f"np.log(pm25_city) ~ {WEATHER_TERMS} + C(season)", data=train.drop(block.index)).fit()
    cv_pred[block.index] = m.predict(block)
print(f"training days: {len(train)} | in-sample R² (log): {model.rsquared:.2f} | "
      f"held-out-month R² (log): {r2_score(np.log(train.pm25_city), cv_pred):.2f}")
print(f"doubling the afternoon mixing height cuts PM2.5 by {1 - 2 ** model.params.lmix:.0%}; "
      f"doubling night mixing height cuts it by a further {1 - 2 ** model.params.lnight:.0%}")

season_window = daily[(daily.md >= 1001) & (daily.md <= 1215) & daily.season.isin(SEASONS)].copy()
season_window["predicted"] = np.exp(model.predict(season_window)) * smear
season_window["excess"] = season_window.pm25_city - season_window.predicted

# %% [markdown]
# **Checks.** The excess should be close to zero *before* the burning starts (early October) and
# *after* it ends (early December, a period the model never saw). It is: −7 and +20 µg/m³. In between
# it rises and falls with the fires.

# %%
windows = [("1–10 Oct", 1001, 1010), ("11–20 Oct", 1011, 1020), ("21–31 Oct", 1021, 1031), ("1–10 Nov", 1101, 1110),
           ("11–20 Nov", 1111, 1120), ("21–30 Nov", 1121, 1130), ("1–15 Dec", 1201, 1215)]
check = pd.DataFrame([{
    "period": name,
    "actual PM2.5": w.pm25_city.mean(),
    "weather-only prediction": w.predicted.mean(),
    "autumn excess": w.excess.mean(),
    "excess share": f"{w.excess.mean() / w.pm25_city.mean():.0%}",
    "fires per day": w.fire_count.mean(),
} for name, a, b in windows for w in [season_window[(season_window.md >= a) & (season_window.md <= b)]]]).set_index("period")
check

# %%
by_day = season_window.groupby("md")[["pm25_city", "predicted", "fire_count"]].mean()
by_day.index = pd.to_datetime([f"2019-{md // 100:02d}-{md % 100:02d}" for md in by_day.index])
smooth = by_day.rolling(5, center=True, min_periods=3).mean()

fig, (ax, axf) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.12})
ax.fill_between(smooth.index, 0, smooth.predicted, color=BLUE_LIGHT, label="Weather-only prediction (Delhi's own pollution)")
ax.fill_between(smooth.index, smooth.predicted, smooth.pm25_city, where=smooth.pm25_city > smooth.predicted,
                color=ORANGE_LIGHT, label="Autumn excess (stubble smoke + Diwali)")
ax.plot(smooth.index, smooth.predicted, color=BLUE, lw=2)
ax.plot(smooth.index, smooth.pm25_city, color=INK, lw=2, label="Actual PM2.5")
ax.axhline(60, color=INK_2, lw=0.8, ls=":")
ax.text(smooth.index[1], 64, "India's 24-hour standard (60)", fontsize=9, color=INK_2)
peak = check.loc["1–10 Nov"]
ax.annotate(f"1–10 Nov: {peak['excess share']} of PM2.5\nis excess the weather can't explain",
            xy=(pd.Timestamp("2019-11-05"), 250), xytext=(pd.Timestamp("2019-11-19"), 300), fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK_2))
ax.set_ylabel("PM2.5 (µg/m³), 5-day average")
ax.set_ylim(0, None)
ax.legend(loc="upper left", fontsize=9.5)
titles(ax, "The smoke arrives with the fires, and leaves with them",
       "Delhi PM2.5 averaged over 2015–2019 by calendar day, against what the weather alone predicts")
axf.bar(by_day.index, by_day.fire_count / 1000, color=ORANGE, width=0.8)
axf.set_ylabel("Fires/day\n(thousands)")
axf.grid(axis="x", visible=False)
axf.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
footnote(fig, "Data: CPCB (Kaggle), NASA FIRMS, Open-Meteo ERA5. Model trained on fire-free months (16 Dec – 31 Mar).", y=0.02)
save(fig, "03_weather_normalised.png")
plt.show()

# %% [markdown]
# ## 4. The smoke fingerprint: it depends on the wind
#
# If the excess really is smoke from Punjab and Haryana, it should be biggest when the wind blows *from*
# there (north-west). The weather model already accounts for what NW winds normally do to Delhi's air in
# winter, so any extra NW effect at the peak of burning season is smoke-specific. Early December, with no
# fires, is the comparison.

# %%
def by_wind(frame):
    frame = frame.copy()
    frame["wind"] = pd.qcut(frame.nw_wind_share, 3, labels=["Mostly other directions", "Mixed", "Mostly north-westerly"])
    return frame.groupby("wind", observed=True).excess.mean()

peak_days = season_window[(season_window.md >= 1021) & (season_window.md <= 1115)]
dec_days = season_window[(season_window.md >= 1201) & (season_window.md <= 1215)]
fingerprint = pd.DataFrame({"Peak burning (21 Oct – 15 Nov)": by_wind(peak_days), "Early December (no fires)": by_wind(dec_days)})
display(fingerprint.T)

fig, ax = plt.subplots(figsize=(9.5, 5.2))
x = np.arange(2)
w = 0.36
for k, (group, color) in enumerate((("Mostly other directions", BLUE), ("Mostly north-westerly", ORANGE))):
    vals = fingerprint.loc[group].values
    b = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=color, label=f"Wind {group.lower()}", edgecolor="#fcfcfb", linewidth=2)
    ax.bar_label(b, labels=[f"{v:+.0f}" for v in vals], padding=3, fontsize=10)
ax.axhline(0, color=INK_2, lw=0.8)
ax.set_xticks(x, fingerprint.columns)
ax.set_ylabel("Autumn excess PM2.5 (µg/m³)")
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right")
titles(ax, "North-westerly winds only add pollution when fields are burning",
       "Excess PM2.5 beyond the weather model, by wind direction (thirds of days by share of hours with NW wind)")
footnote(fig, "Data: CPCB (Kaggle), Open-Meteo ERA5, 2015–2019.", y=-0.04)
save(fig, "04_wind_fingerprint.png")
plt.show()

# %% [markdown]
# ## 5. Diwali: one of the worst nights of every year, but a short spike

# %%
events = []
for d in DIWALI:
    w = hourly[(hourly.datetime >= d - pd.Timedelta(days=2)) & (hourly.datetime < d + pd.Timedelta(days=4))].copy()
    w["hours"] = (w.datetime - d) / pd.Timedelta(hours=1)
    events.append(w.assign(year=d.year))
events = pd.concat(events)
mean_curve = events.groupby("hours").pm25.mean()

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axvspan(18, 30, color="#ebe9f6", zorder=0)
ax.text(24, 30, "Diwali night\n6 pm – 6 am", ha="center", fontsize=9.5, color=VIOLET)
for year, g in events.groupby("year"):
    ax.plot(g.hours, g.pm25, color=NEUTRAL, lw=1)
    night = g[(g.hours >= 18) & (g.hours < 30)]
    top = night.loc[night.pm25.idxmax()]
    ax.text(top.hours + 1, top.pm25, str(year), fontsize=8.5, color=INK_2, va="center")
ax.plot(mean_curve.index, mean_curve.values, color=VIOLET, lw=2.5, label="Average of 5 Diwalis")
DAY_NAMES = ["D−2", "D−1", "Diwali", "D+1", "D+2", "D+3", "D+4"]
ax.set_xticks(range(-48, 97, 12), [f"00:00\n{DAY_NAMES[(h + 48) // 24]}" if h % 24 == 0 else "12:00" for h in range(-48, 97, 12)], fontsize=8.5)
ax.set_xlim(-48, 96)
ax.set_ylim(0, None)
ax.set_ylabel("Hourly PM2.5 (µg/m³)")
ax.legend(loc="upper left")
night_peaks = events[(events.hours >= 18) & (events.hours < 30)].groupby("year").pm25.max()
titles(ax, "Diwali night: one of the five worst nights of every year",
       f"Hourly Delhi PM2.5 around Diwali, 2015–2019. Night peaks of {night_peaks.min():.0f}–{night_peaks.max():.0f} µg/m³ "
       "(WHO 24-hour guideline: 15). In this season every night is bad.")
footnote(fig, "Data: CPCB city-average hourly PM2.5 (Kaggle).", y=-0.05)
save(fig, "05_diwali_hourly.png")
plt.show()

# how Diwali night ranks among each year's 365 daily peaks
ranks = []
for d in DIWALI:
    year = hourly[hourly.datetime.dt.year == d.year].dropna()
    daily_peak = year.groupby(year.datetime.dt.date).pm25.max()
    ranks.append({"Diwali": d.date(), "night peak (µg/m³)": night_peaks[d.year],
                  "rank among the year's daily peaks": int((daily_peak > night_peaks[d.year]).sum()) + 1})
pd.DataFrame(ranks).set_index("Diwali")

# %% [markdown]
# **How big is Diwali's share?** Pollution keeps climbing for days after Diwali, because Diwali falls
# inside the burning season. So the comparison is the weather-normalised excess on Diwali day and the
# day after, against the excess on the three days before *and* after.

# %%
excess_by_date = season_window.set_index("date").excess
bumps = []
for d in DIWALI:
    around = np.nanmean([excess_by_date.get(d + pd.Timedelta(days=k), np.nan) for k in (-3, -2, -1, 2, 3, 4)])
    bump = sum(excess_by_date.get(d + pd.Timedelta(days=k)) - around for k in (0, 1))
    bumps.append({"Diwali": d.date(), "excess on Diwali + next day": excess_by_date.get(d) + excess_by_date.get(d + pd.Timedelta(days=1)),
                  "typical excess around it (×2 days)": 2 * around, "Diwali bump (µg/m³·days)": bump})
bumps = pd.DataFrame(bumps).set_index("Diwali")
diwali_bump = bumps["Diwali bump (µg/m³·days)"].mean()
bumps

# %% [markdown]
# ## 6. The pollution budget
#
# Adding it up for each winter (15 October – 28 February), how much of the total PM2.5 exposure comes
# from each source?

# %%
budget = []
for s in SEASONS:
    winter = daily[(daily.date >= f"{s}-10-15") & (daily.date <= f"{s + 1}-02-28")]
    total = winter.pm25_city.mean() * len(winter)                       # µg/m³·days
    autumn_excess = season_window[(season_window.season == s) & (season_window.md >= 1015) & (season_window.md <= 1130)].excess.sum()
    diwali = bumps["Diwali bump (µg/m³·days)"].iloc[s - 2015]
    budget.append({"season": s, "Diwali": diwali / total, "Stubble-season smoke": (autumn_excess - diwali) / total,
                   "Delhi's own pollution + weather": 1 - autumn_excess / total})
budget = pd.DataFrame(budget).set_index("season")

nov = season_window[(season_window.md >= 1101) & (season_window.md <= 1110)
                    & ~season_window.date.isin(DIWALI) & ~season_window.date.isin(DIWALI + pd.Timedelta(days=1))]
peak_share = nov.excess.mean() / nov.pm25_city.mean()
shares = pd.DataFrame({
    "Whole winter\n(15 Oct – 28 Feb)": budget.mean(),
    "Peak smoke\n(1–10 Nov, excl. Diwali)": pd.Series({"Diwali": 0, "Stubble-season smoke": peak_share,
                                                     "Delhi's own pollution + weather": 1 - peak_share}),
}).T
display((100 * budget).round(1))

fig, ax = plt.subplots(figsize=(11, 4.2))
colors = {"Delhi's own pollution + weather": BLUE, "Stubble-season smoke": ORANGE, "Diwali": VIOLET}
left = np.zeros(len(shares))
for part in ("Delhi's own pollution + weather", "Stubble-season smoke", "Diwali"):
    vals = shares[part].values
    ax.barh(range(len(shares)), vals, left=left, color=colors[part], height=0.55, edgecolor="#fcfcfb", linewidth=2, label=part)
    for i, (v, l) in enumerate(zip(vals, left)):
        if v >= 0.06:
            ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    left += vals
ax.annotate(f"Diwali: {shares.iloc[0]['Diwali']:.1%}", xy=(1.0, 0), xytext=(1.02, -0.42), fontsize=10, color=VIOLET,
            arrowprops=dict(arrowstyle="-", color=VIOLET))
ax.set_yticks(range(len(shares)), shares.index)
ax.tick_params(axis="y", length=0)
ax.invert_yaxis()
ax.set_xlim(0, 1.12)
ax.set_xticks([])
ax.grid(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.05), ncol=3)
titles(ax, "Stubble smoke owns early November. Delhi's own pollution owns the winter.",
       "Share of PM2.5 exposure by source, average of 2015–2019 (weather-normalised estimate)")
footnote(fig, "Method: excess PM2.5 beyond a weather model trained on fire-free months. Data: CPCB, NASA FIRMS, Open-Meteo.", y=-0.12)
save(fig, "06_pollution_budget.png")
plt.show()

# %% [markdown]
# ## 7. Robustness
#
# The headline numbers shouldn't depend on one modelling choice. Each variant is re-run below.

# %%
def run_variant(y="pm25_city", train_mask=None, formula=None, gbm=False):
    target = daily[(daily.md >= 1001) & (daily.md <= 1215) & daily.season.isin(SEASONS)].copy()
    if gbm:
        cols = ["wind_kmh", "wind_u", "wind_v", "lmix", "lnight", "temp_c", "temp_range_c", "rh_pct", "rain_mm", "season", "dow"]
        tr = daily[((daily.md >= 1216) | (daily.md <= 331)) & ~covid & daily[y].notna()]
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=30, random_state=0)
        g.fit(tr[cols], np.log(tr[y]))
        sm = np.mean(np.exp(np.log(tr[y]) - g.predict(tr[cols])))
        target["predicted"] = np.exp(g.predict(target[cols])) * sm
    else:
        m, sm, _ = weather_model(y, train_mask, formula)
        target["predicted"] = np.exp(m.predict(target)) * sm
    target["excess"] = target[y] - target.predicted
    win = lambda a, b: target[(target.md >= a) & (target.md <= b)]
    burn = win(1015, 1130)
    return {"early Oct excess (should be ~0)": win(1001, 1010).excess.mean(),
            "15 Oct–30 Nov excess": burn.excess.mean(),
            "share of PM2.5": f"{burn.excess.mean() / burn[y].mean():.0%}",
            "1–10 Nov excess": win(1101, 1110).excess.mean(),
            "early Dec excess (should be ~0)": win(1201, 1215).excess.mean()}

pd.DataFrame({
    "Main model": run_variant(),
    "10 long-running stations only": run_variant("pm25_core10"),
    "Train on Jan–Mar only": run_variant(train_mask=(daily.md >= 101) & (daily.md <= 331)),
    "Season as a trend, not levels": run_variant(formula=f"np.log(pm25_city) ~ {WEATHER_TERMS} + season"),
    "Gradient boosting model": run_variant(gbm=True),
}).T

# %% [markdown]
# Every variant puts the autumn excess at **85–100 µg/m³, or 40–46% of PM2.5, from mid-October to the end
# of November**. Early October stays within 20 µg/m³ of zero in every case. Early December sits at +20 to +26, which
# suggests the model slightly under-predicts in late autumn. Subtracting that bias from the main estimate
# gives a lower bound of about a third of PM2.5.

# %% [markdown]
# ## Limitations
#
# - **The excess is "autumn-only pollution", not proven stubble smoke.** Its timing matches the fires, and
#   it depends on wind from the fire region, but other autumn-only sources (Dussehra, regional burning
#   outside Punjab and Haryana) are mixed in.
# - **The PM2.5 data ends in mid-2020**, so it can't show whether Delhi's air improved as fire detections fell
#   from 2021 to 2025. Satellite counts may also understate burning if farmers burn after the satellite passes.
# - **Five seasons is a small sample.** Year-to-year numbers vary (Diwali's bump ranges from 13 to 286 µg/m³·days).
# - **The weather comes from one grid point** (ERA5 reanalysis at central Delhi), not local measurements.

# %%
season_window[["date", "pm25_city", "predicted", "excess", "fire_count", "nw_wind_share", "mixing_height_m"]].to_csv(
    PROC / "weather_normalised_daily.csv", index=False)
