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
| 4 | Batch inference → dashboard | 🔄 In progress |
| 5 | Hindcast graph (monitoring) | ✅ Done |
| 6 | Lagged features (Grade C) | ⬜ Not started |
| 7 | All sensors in one city (Grade A) | ⬜ Not started |

---

## Task 1 — Backfill feature pipeline

Historical air quality loaded from an aqicn.org CSV export; historical weather
pulled from the Open-Meteo archive API for the sensor's coordinates.

| Feature Group | Rows | Features | Time range |
|---|---|---|---|
| `air_quality` v1 | 3,915 | `date`, `pm25`, `country`, `city`, `street`, `url` | 2014-01-31 → 2026-08-07 |
| `weather` v1 | 4,571 | `date`, `temperature_2m_mean`, `precipitation_sum`, `wind_speed_10m_max`, `wind_direction_10m_dominant`, `city` | 2014-01-31 → 2026-08-06 |

**≈12.5 years of history**, well beyond the ">1 year" the task asks for.

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

Hindcast chart of predictions vs measured outcomes over the 61-day test window,
saved with the model in the registry and published to the dashboard. The
`aq_predictions` feature group records each day's forecast so that prediction
accuracy can be tracked as outcomes arrive.

---

## Task 4 — Batch inference & dashboard

_(to fill in after the run)_

| | |
|---|---|
| Forecast days | |
| Predicted PM2.5 range | |
| Dashboard URL | |

---

## Task 6 — Lagged features (Grade C)

_(to fill in)_

| Model | Features | MSE | RMSE | R² |
|---|---|---|---|---|
| v1 baseline | 4 weather | 178.94 | 13.38 | −0.533 |
| v2 + lags | 4 weather + `pm25_lag_1/2/3` | | | |

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
