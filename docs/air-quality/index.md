# Air Quality Prediction Service

**PM2.5 forecasts for Olivais, Lisboa, Portugal**
Sensor: [AQICN station 10513](https://aqicn.org/station/portugal/lisboa/olivais/) · 38.7689 °N, −9.1081 °E

A serverless ML system that forecasts fine-particulate pollution for the next 7
days. Weather forecasts come from [Open-Meteo](https://open-meteo.com), air
quality measurements from [AQICN](https://aqicn.org), features are stored in
[Hopsworks](https://www.hopsworks.ai), and the pipelines run daily on GitHub
Actions.

{% include air-quality.html %}

## 7-Day PM2.5 Forecast

![Forecast](./assets/img/pm25_forecast.png)

Predictions come from an XGBoost regressor trained on ~12 years of daily
observations, using temperature, precipitation, wind speed and wind direction
as features.

## Model Performance Monitoring

1-Day Hindcast: predictions vs measured outcomes

![Hindcast](./assets/img/pm25_hindcast_1day.png)

Every forecast is written to a monitoring feature group alongside the horizon it
was made at, so predictions can be compared against the outcomes that arrive
later. The chart above shows only forecasts made one day ahead.

---

*Built for ID2223 (Scalable Machine Learning and Deep Learning), KTH.
[Source code](https://github.com/margaridagsousa/Lab1mlfs-book).*
