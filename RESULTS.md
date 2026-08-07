# Lab 1 — Results Log

Running record of what was built and measured, for the report and the defence.

**Project:** ID2223 Lab 1 — Air Quality Prediction Service
**Sensor:** Olivais, Lisboa, Portugal — [AQICN station 10513](https://api.waqi.info/feed/@10513)
**Coordinates:** 38.7689 °N, −9.1081 °E
**Hopsworks project:** `id2223_lab1_airquality`
**Repository:** https://github.com/margaridagsousa/Lab1mlfs-book

---

## Task status

| # | Task | Status |
|---|------|--------|
| 1 | Backfill pipeline → 2 Feature Groups | ✅ Done |
| 2 | Daily feature pipeline on GitHub Actions | ✅ Done |
| 3 | Training pipeline → registered model | ✅ Done |
| 4 | Batch inference → dashboard | ✅ Done |
| 5 | Hindcast graph (monitoring) | ✅ Done |
| 6 | Lagged features (Grade C) | ✅ Done |
| 7 | All sensors in one city (Grade A) | 🔄 In progress |

---

## Task 1 — Backfill feature pipeline

Historical air quality loaded from an aqicn.org CSV export; historical weather
pulled from the Open-Meteo archive API for the sensor's coordinates.

| Feature Group | Rows | Features | Time range |
|---|---|---|---|
| `air_quality` v1 | 3,915 | `date`, `pm25`, `country`, `city`, `street`, `url` | 2014-01-31 → 2026-08-07 |
| `weather` v1 | 4,571 | `date`, `temperature_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`, `wind_direction_10m_dominant`, `city` | 2014-01-31 → 2026-08-06 |

**≈12.5 years of history**, well beyond the ">1 year" the task asks for.

A third feature group, `aq_predictions` v1, is created later by the batch
inference pipeline (Task 4) to store predictions for monitoring.

Design notes worth defending:
- `air_quality` is keyed on `['country','city','street']` — a **specific sensor**.
- `weather` is keyed on `['city']` — weather is a **city-wide** property, so all
  sensors in Lisboa share one weather row per day.
- Both use `event_time='date'`, which is what makes the point-in-time join and
  the date-based train/test split possible.
- Weather has ~650 more rows than air quality because the sensor has gaps on
  days the weather archive still covers. The join keeps only overlapping dates.

---

## Task 2 — Daily feature pipeline (GitHub Actions)

`.github/workflows/air-quality-daily.yml` — scheduled `cron: '11 6 * * *'`
(06:11 UTC daily) plus manual `workflow_dispatch`.

Each run: reads yesterday's PM2.5 from the AQICN API, pulls the **7-day ECMWF
weather forecast** from Open-Meteo, and inserts both into the feature groups.

- First green run: **#4**, 4m 32s, commit `86372cb`
- Forecast horizon measured: **168 hourly rows = 7 days** (task asks 7–10)

---

## Task 3 — Training pipeline

**Feature View:** `air_quality_fv` v1 — `air_quality.pm25` as the label, joined
to the weather features on `city`, aligned by date.

**Model:** XGBoost regressor (`XGBRegressor`, default hyperparameters),
registered as `air_quality_xgboost_model` v1.

### Split

Split **by date**, not randomly — a random split over a time series leaks
future days into training and inflates the score.

| | Rows | Period |
|---|---|---|
| Train | 3,855 | 2014-01-31 → 2026-06-07 |
| Test | 61 | 2026-06-08 → 2026-08-07 (last 60 days) |

### Baseline results — weather features only

| Metric | Value |
|---|---|
| MSE | **178.94** |
| RMSE | **13.38 µg/m³** |
| R² | **−0.533** |

**Features used:** `temperature_2m_mean`, `precipitation_sum`,
`wind_speed_10m_max`, `wind_direction_10m_dominant`

### Feature importance

| Feature | Importance (weight) |
|---|---|
| `temperature_2m_mean` | 1552 |
| `wind_speed_10m_max` | 1357 |
| `wind_direction_10m_dominant` | 1242 |
| `precipitation_sum` | 462 |

### Interpreting a negative R²

R² = −0.533 means the model performs **worse than predicting the test-set mean
every day**. This is a genuine result, not a defect, and it is the interesting
finding of the baseline:

1. **Weather alone is weak evidence for PM2.5.** Pollution is produced by
   traffic and industry; weather mostly *disperses* it. The four available
   features describe dispersion, not emission.
2. **`temperature_2m_mean` dominating is a warning sign** — temperature largely
   encodes *season*, so the model is leaning on a seasonal proxy rather than
   anything causal.
3. **Train/test distribution mismatch.** Training spans 12 years across all
   seasons; the test window is 61 summer days sitting in a narrow 17–58 µg/m³
   band. The model over-predicts, swinging up to ~70 where actuals stay ≤58.
4. **The strongest available signal is missing:** yesterday's PM2.5. Air
   quality is highly autocorrelated — this is exactly what Task 6 adds.

This baseline is the "before" half of the Task 6 comparison.

---

## Task 5 — Monitoring / hindcast

Hindcast chart of predictions vs measured outcomes, published to the dashboard
at `docs/air-quality/assets/img/pm25_hindcast_1day.png`.

The `aq_predictions` feature group records every forecast together with the
horizon it was made at (`days_before_forecast_day`), so predictions can be
compared against outcomes as they arrive. The chart selects only forecasts made
1 day ahead.

### Hindcast over the 61-day test window

| Metric | Value |
|---|---|
| Days compared | 61 (2026-06-08 → 2026-08-07) |
| MSE | 178.94 |
| R² | −0.533 |
| **Mean bias** | **+6.41 µg/m³ (over-predicting)** |

MSE and R² match the Task 3 test metrics exactly, confirming the hindcast
replays the same held-out window.

### What the chart shows

Two failure modes are visible and worth pointing out:

1. **Systematic over-prediction** — the predicted line sits above the actual
   line on most days, quantified as the +6.41 µg/m³ mean bias. Several days are
   pushed into the "Moderate" band when the true reading was "Good".
2. **Excess day-to-day volatility** — measured PM2.5 moves in smooth multi-day
   waves, while predictions zigzag. The model is tracking daily weather noise,
   but real pollution has *momentum* that weather features cannot express.

That momentum is exactly what the lagged features in Task 6 supply.

### Note on how the hindcast was populated

The daily pipeline only accumulates predictions from the day it first runs, so a
fresh deployment has a single matched pair. The monitoring group was therefore
backfilled by replaying the trained model over the past 60 days of weather
already in the feature store.

These remain honest out-of-sample predictions: the model was trained on data up
to 2026-06-07, so every backfilled day was unseen during training.

---

## Task 4 — Batch inference & dashboard

Downloads `air_quality_xgboost_model` v1 from the Hopsworks model registry,
reads the weather **forecast** rows written by the daily pipeline, and predicts
PM2.5 for each of the next 7 days.

| | |
|---|---|
| Forecast days | 7 (2026-08-07 → 2026-08-13) |
| Predicted PM2.5 range | 26.5 – 71.4 µg/m³ |
| Dashboard image | `docs/air-quality/assets/img/pm25_forecast.png` |
| Dashboard URL | https://margaridagsousa.github.io/Lab1mlfs-book/air-quality/ |

### Forecast produced 2026-08-07

| Date | Predicted PM2.5 (µg/m³) | AQI band |
|---|---|---|
| 2026-08-07 | 52.3 | Moderate |
| 2026-08-08 | 57.7 | Moderate |
| 2026-08-09 | 71.4 | Moderate |
| 2026-08-10 | 69.8 | Moderate |
| 2026-08-11 | 55.9 | Moderate |
| 2026-08-12 | 26.5 | Good |
| 2026-08-13 | 56.2 | Moderate |

A third feature group, `aq_predictions` v1, keyed on
`['city','street','date','days_before_forecast_day']`, stores every forecast so
that predictions can later be compared against measured outcomes. The
`days_before_forecast_day` column records the forecast horizon, which is what
makes a *1-day* hindcast selectable from among all stored predictions.

### Caveat on these predictions

The baseline model over-predicts (see Task 3): its test-window predictions
reached ~70 µg/m³ where actuals stayed ≤58. The forecast above should be read
with that bias in mind — the 71.4 µg/m³ peak is more likely mid-50s in reality.
Task 6 addresses this.

---

## Task 6 — Lagged features (Grade C)

Added `pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3` — the measured PM2.5 from 1, 2
and 3 days earlier — as features, and measured the effect.

**Hypothesis:** PM2.5 is strongly autocorrelated, and the baseline model had no
access to that. Weather *disperses* pollution rather than producing it, so
yesterday's reading should be the single most informative feature available.

Measured autocorrelation at this sensor:

| Lag | Correlation with today's PM2.5 |
|---|---|
| 1 day | **+0.676** |
| 2 days | +0.428 |
| 3 days | +0.308 |

### Results

Both models trained and tested on **identical rows** (train 3,415 / test 61,
2026-06-08 → 2026-08-07), so the difference is attributable to the features
alone and not to a different split.

| Model | Features | MSE | RMSE | R² |
|---|---|---|---|---|
| v1 weather only | 4 weather | 222.15 | 14.90 | −0.904 |
| **v2 weather + lags** | 4 weather + 3 lags | **56.73** | **7.53** | **+0.514** |
| Persistence baseline | `pm25_lag_1` alone | 45.66 | 6.76 | +0.609 |

**MSE improved by 74.5%.** R² moved from −0.904 (worse than predicting the
mean) to +0.514 (explaining about half the variance). Typical daily error
halved, from 14.9 to 7.5 µg/m³.

Registered as `air_quality_xgboost_model_lagged` v1.

> Note: v1 scores differently here (MSE 222.15) than in Task 3 (MSE 178.94)
> because building the lags drops the first days and any row following a gap in
> the sensor record. v1 was therefore re-trained on this smaller, identical
> subset so the comparison is fair.

### The persistence baseline beats the model

Predicting "today = yesterday" scores **MSE 45.66 / R² +0.609**, better than
the trained model's 56.73 / +0.514. This is the most important finding of the
task and should be stated plainly rather than hidden:

1. **Almost all the improvement came from the lag features, not from the
   model.** Once the model can see yesterday's value, it is essentially
   learning a noisy approximation of "repeat yesterday".
2. **The weather features actively hurt.** They inject day-to-day variance the
   target does not have, so the model oscillates around the persistence signal
   instead of tracking it. This is visible in the comparison chart, where the
   red v1 line spikes to 86 µg/m³ on a day the actual reading was ~30.
3. **A good baseline is essential.** Reporting only "MSE improved 74.5%" would
   have looked like a success story, while the honest conclusion is that a
   one-line heuristic still outperforms the ML model on this problem.

### Feature importance for v2

| Feature | Importance |
|---|---|
| `temperature_2m_mean` | 1060 |
| `wind_speed_10m_max` | 793 |
| `wind_direction_10m_dominant` | 782 |
| `pm25_lag_1` | 612 |
| `pm25_lag_2` | 572 |
| `pm25_lag_3` | 550 |
| `precipitation_sum` | 322 |

XGBoost's default importance counts *how often* a feature is split on, not how
much predictive value it carries. Weather features are continuous and
high-cardinality, so they attract many low-value splits; the lags are used less
often but carry far more signal — as the 74.5% MSE improvement shows. Worth
knowing at the defence: this metric is easy to misread.

### How the lags were computed

The sensor record has **656 missing days** out of a 4,571-day span. A naive
`shift(1)` over the rows present would have paired readings across those gaps —
175 rows affected, the worst pairing readings **49 days apart** as "yesterday".

The lags are therefore computed after reindexing to a continuous daily calendar,
leaving gaps as NaN, and rows without a complete set of lags are dropped. That
keeps 3,476 of 3,916 readings (88.8%) and guarantees every lag value really is
from the previous day.

---

## Task 7 — All sensors in Lisboa (Grade A)

Lisboa has **3 AQICN sensors**:

| Station | uid | Coordinates |
|---|---|---|
| Olivais, Lisboa | 10513 | 38.7689, −9.1081 |
| Entrecampos, Lisboa | 8379 | 38.7486, −9.1489 |
| Laranjeiro, Almada | 8381 | 38.6636, −9.1578 |

_(to fill in)_

---

## Engineering problems solved

Worth mentioning at the defence — these are real production concerns, not
incidental noise.

| Problem | Cause | Resolution |
|---|---|---|
| `hopsworks` will not install on Windows | `pyjks` → `twofish` has no Windows wheel and needs a C++ compiler; WSL unavailable | Ran the pipelines in Google Colab (Linux) |
| `JSONDecodeError` on login | `app.hopsworks.ai` serves the web UI, not the API | Used `eu-west.cloud.hopsworks.ai` |
| `DataContextRequiredError` | Great Expectations 1.x requires a DataContext; Hopsworks targets the 0.18.x API | Made validation optional; assert PM2.5 range directly |
| `Cannot use time_travel_format='DELTA'` | `hsfs` caches library availability at import; `deltalake` was installed afterwards | Install before the first `hopsworks` import |
| `RPC listener disconnected` on insert | DELTA writes go via `delta-rs` straight to Hopsworks HDFS, which fails from Colab | Created the feature groups with `time_travel_format='HUDI'` |
| `ImportError: cannot import name 'util'` in CI | Notebooks derive the repo root from the cwd; running from the repo root resolved `airquality` to the outer directory | Set `working-directory: airquality/notebooks` |
| `NameError: RequestsCookieJar` in CI | `requests-cache` annotations fail to resolve on Python 3.12 against newer `requests` | Replaced `CachedSession` with a plain retry session |
