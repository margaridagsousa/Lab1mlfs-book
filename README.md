# ID2223 Lab 1 — Air Quality Prediction Service

A serverless ML system that forecasts PM2.5 air pollution for **every air
quality sensor in the Lisboa area**, built on a feature store with pipelines
that run daily without a server.

**📊 Dashboard:** https://margaridagsousa.github.io/Lab1mlfs-book/air-quality/

**📈 Detailed results and analysis:** [RESULTS.md](RESULTS.md)

Author: Margarida Sousa · KTH, ID2223 Scalable Machine Learning and Deep Learning

---

## What it does

Predicts the daily PM2.5 level for the next 7 days at three AQICN monitoring
stations, from weather forecasts and recent pollution history.

| Station | AQICN uid | Coordinates |
|---|---|---|
| Olivais, Lisboa | [10513](https://aqicn.org/station/portugal/lisboa/olivais/) | 38.7689 °N, −9.1081 °E |
| Entrecampos, Lisboa | [8379](https://aqicn.org/station/portugal/lisboa/entrecampos/) | 38.7486 °N, −9.1489 °E |
| Laranjeiro, Almada | [8381](https://aqicn.org/station/portugal/almada/laranjeiro/) | 38.6636 °N, −9.1578 °E |

Data sources: [AQICN](https://aqicn.org) for measured air quality,
[Open-Meteo](https://open-meteo.com) for historical weather and ECMWF forecasts.

## Architecture

Feature/Training/Inference (FTI) pipelines around a
[Hopsworks](https://www.hopsworks.ai) feature store:

```
   AQICN API ─┐
              ├──► backfill + daily feature pipelines ──► Hopsworks Feature Store
Open-Meteo API ┘                                              │
                                                              ├──► training pipeline ──► Model Registry
                                                              │                              │
                                                              └──► batch inference ◄─────────┘
                                                                        │
                                                                        └──► GitHub Pages dashboard
```

**Feature groups**

| Name | Primary key | Contents |
|---|---|---|
| `air_quality` | `country`, `city`, `street` | Daily PM2.5 per sensor |
| `weather` | `city` | Daily temperature, precipitation, wind speed, wind direction |
| `air_quality_lagged` | `country`, `city`, `street` | PM2.5 plus 1/2/3-day lags |
| `air_quality_all_sensors` | `country`, `city`, `street` | All three sensors, with lags |
| `aq_predictions` | `city`, `street`, `date`, `days_before_forecast_day` | Stored forecasts, for monitoring |

Air quality is keyed **per sensor** while weather is keyed **per city** — the
sensors are a few km apart and share one weather record per day. That is why
extending from one sensor to three needed no new weather data at all.

`event_time='date'` on every group is what makes the point-in-time join and the
date-based train/test split possible.

## Results

| Model | Features | MSE | RMSE | R² |
|---|---|---|---|---|
| Baseline | 4 weather | 222.15 | 14.90 | −0.904 |
| **+ lagged PM2.5** | 4 weather + 3 lags | **56.73** | **7.53** | **+0.514** |
| All sensors | + `street` | 67.87 | 8.24 | +0.521 |
| *Persistence baseline* | *`pm25_lag_1` alone* | *45.66* | *6.76* | *+0.609* |

Three findings worth highlighting:

1. **Weather alone cannot predict PM2.5.** The baseline scored R² = −0.904 —
   worse than predicting the mean. Weather disperses pollution; it does not
   create it.
2. **Lagged features cut MSE by 74.5%.** PM2.5 correlates +0.68 with the
   previous day, and that autocorrelation is the strongest available signal.
3. **A persistence baseline still beats the trained model.** "Today equals
   yesterday" scores MSE 45.66 against the model's 56.73. Most of the
   improvement in (2) comes from the lag features rather than from the
   learning, and the weather features add variance the target does not have.
   Reporting only the 74.5% improvement would have been misleading.

See [RESULTS.md](RESULTS.md) for per-sensor breakdowns, the hindcast analysis,
and a discussion of why multi-step forecasts compound bias.

## Repository layout

```
airquality/notebooks/
  1_air_quality_feature_backfill.ipynb   original book notebooks,
  2_air_quality_feature_pipeline.ipynb   retargeted to the Olivais sensor
  3_air_quality_training_pipeline.ipynb
  4_air_quality_batch_inference.ipynb

  colab_1_backfill.ipynb                 Task 1  backfill
  colab_3_training.ipynb                 Task 3  training + Task 5 hindcast
  colab_4_inference.ipynb                Task 4  batch inference + dashboard
  colab_4b_hindcast_backfill.ipynb       Task 5  populate the hindcast
  colab_6_lagged_features.ipynb          Task 6  lagged features (Grade C)
  colab_7_all_sensors.ipynb              Task 7  all sensors (Grade A)

.github/workflows/air-quality-daily.yml  daily pipeline, 06:11 UTC
data/                                    historical CSVs from aqicn.org
docs/air-quality/                        the published dashboard
RESULTS.md                               measurements and analysis
```

The `colab_*` notebooks are self-contained versions written for Google Colab —
see [Running the pipelines](#running-the-pipelines) below for why.

## Running the pipelines

The daily feature pipeline runs automatically on GitHub Actions
(`cron: '11 6 * * *'`) and can also be triggered manually from the Actions tab.
It needs four repository secrets: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT`,
`HOPSWORKS_HOST` and `AQICN_API_KEY`.

The other pipelines are run from the `colab_*` notebooks. They exist because
`hopsworks` cannot be installed on native Windows — its `pyjks` → `twofish`
dependency has no Windows wheel and needs a C++ toolchain — so the pipelines
were developed and run on Linux via Colab. Each notebook is self-contained:
it installs its dependencies, prompts for credentials, and needs no local
checkout.

For a local Linux/macOS/WSL setup instead:

```bash
cp .env.example .env       # then fill in your keys
cd airquality
source setup.sh
inv all
```

## Notes on the implementation

A few decisions that differ from the book's version, each made in response to a
concrete failure — the full list is in the last section of [RESULTS.md](RESULTS.md):

- **Feature groups use `time_travel_format='HUDI'`.** The cluster defaults to
  DELTA, whose writes go through `delta-rs` directly to Hopsworks HDFS and fail
  from Colab with `RPC listener disconnected`.
- **`requests-cache` was removed** from the Open-Meteo clients. On Python 3.12
  its annotations fail to resolve against newer `requests`, breaking the daily
  pipeline in CI with `NameError: RequestsCookieJar`.
- **Lags are computed on a continuous daily calendar**, not with a plain
  `shift()`. The sensor record has 656 missing days, and shifting over present
  rows alone would have paired readings up to 49 days apart as "yesterday".
- **Great Expectations validation is optional.** GE 1.x requires an active
  DataContext to build a suite, while the Hopsworks integration targets the
  0.18.x API; the PM2.5 range is asserted directly instead.

## Acknowledgements

Built from the example project in *Building Machine Learning Systems with a
Feature Store* by Jim Dowling
([featurestorebook/mlfs-book](https://github.com/featurestorebook/mlfs-book)).
