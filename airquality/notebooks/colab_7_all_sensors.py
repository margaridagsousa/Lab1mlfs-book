# ============================================================
# TASK 7 (Grade A) - predictions for every sensor in the city
#
# Extends the system from one sensor to all three AQICN stations in the
# Lisboa area, end to end: backfill, lags, one model across sensors,
# and a dashboard per sensor.
#
# Upload all three CSVs to the Colab session first:
#   olivais-lisboa.csv, entrecampos-lisboa.csv, laranjeiro-almada.csv
#
# Install first on a fresh runtime:
#   !pip install -q hopsworks confluent-kafka deltalake xgboost \
#       scikit-learn matplotlib
# ============================================================
import getpass, os, json, datetime
import pandas as pd
import numpy as np

os.environ["HOPSWORKS_API_KEY"] = getpass.getpass("HOPSWORKS_API_KEY: ").strip()
os.environ["HOPSWORKS_PROJECT"] = input("HOPSWORKS_PROJECT: ").strip()
os.environ["HOPSWORKS_HOST"]    = "eu-west.cloud.hopsworks.ai"
AQICN_API_KEY = getpass.getpass("AQICN_API_KEY: ").strip()

import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
print("connected to", project.name)

# --- the sensors -------------------------------------------------
# uid and coordinates come from https://api.waqi.info/search/?keyword=lisboa
SENSORS = [
    {"street": "Olivais",     "city": "Lisboa", "country": "Portugal",
     "uid": 10513, "lat": 38.76888900025,  "lon": -9.108055999671,
     "csv": "olivais-lisboa.csv"},
    {"street": "Entrecampos", "city": "Lisboa", "country": "Portugal",
     "uid": 8379,  "lat": 38.748611111111, "lon": -9.1488888888889,
     "csv": "entrecampos-lisboa.csv"},
    {"street": "Laranjeiro",  "city": "Lisboa", "country": "Portugal",
     "uid": 8381,  "lat": 38.663611111111, "lon": -9.1577777777778,
     "csv": "laranjeiro-almada.csv"},
]
for s in SENSORS:
    s["aqicn_url"] = f"https://api.waqi.info/feed/@{s['uid']}"

CITY = "Lisboa"
LAGS = [1, 2, 3]
LAG_FEATURES = [f'pm25_lag_{l}' for l in LAGS]
WEATHER_FEATURES = ['temperature_2m_mean', 'precipitation_sum',
                    'wind_speed_10m_max', 'wind_direction_10m_dominant']
TEST_DAYS = 60

# --- 1. load every sensor's history, with lags -------------------
def load_sensor(s):
    df = pd.read_csv(s['csv'], parse_dates=['date'], skipinitialspace=True)
    df = df[['date', 'pm25']].copy()
    df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
    df = df.dropna(subset=['pm25']).sort_values('date')

    # Lags on a continuous calendar, so gaps in the record never get
    # silently treated as consecutive days.
    full = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
    daily = df.set_index('date').reindex(full)
    daily.index.name = 'date'
    for l in LAGS:
        daily[f'pm25_lag_{l}'] = daily['pm25'].shift(l)

    out = daily.reset_index().dropna(subset=['pm25'] + LAG_FEATURES)
    out['country'], out['city'], out['street'] = s['country'], s['city'], s['street']
    out['url'] = s['aqicn_url']
    for c in ['pm25'] + LAG_FEATURES:
        out[c] = out[c].astype('float32')
    return out

frames = []
for s in SENSORS:
    d = load_sensor(s)
    frames.append(d)
    print(f"  {s['street']:14} {len(d):5,} rows  "
          f"{d['date'].min().date()} -> {d['date'].max().date()}  "
          f"mean pm25={d['pm25'].mean():.1f}")

all_aq = pd.concat(frames, ignore_index=True)
print(f"\ntotal rows across {len(SENSORS)} sensors: {len(all_aq):,}")

# --- 2. one feature group holding every sensor -------------------
# The primary key already includes street, so several sensors coexist in
# one feature group without any schema change.
aq_all_fg = fs.get_or_create_feature_group(
    name='air_quality_all_sensors',
    description=f'Air quality with 1/2/3-day lags for all sensors in {CITY}',
    version=1,
    primary_key=['country', 'city', 'street'],
    event_time='date',
    time_travel_format='HUDI',
)
aq_all_fg.insert(
    all_aq[['date', 'pm25'] + LAG_FEATURES + ['country', 'city', 'street']],
    wait=True)
print("air_quality_all_sensors written")

# --- 3. feature view across all sensors --------------------------
weather_fg = fs.get_feature_group(name='weather', version=1)

selected = aq_all_fg.select(['pm25', 'date', 'street'] + LAG_FEATURES).join(
    weather_fg.select_features(), on=['city'])

fv = fs.get_or_create_feature_view(
    name='air_quality_fv_all_sensors',
    description=f'All {CITY} sensors: weather + lagged pm25',
    version=1,
    labels=['pm25'],
    query=selected,
)
print("feature view ready: air_quality_fv_all_sensors v1")

# --- 4. one model across all sensors -----------------------------
# `street` enters as a categorical feature so a single model can learn
# each site's offset (the three sensors differ by several ug/m3) while
# sharing everything they have in common.
df = fv.query.read()
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
df = df.sort_values('date').reset_index(drop=True)
df['street'] = df['street'].astype('category')

test_start = df['date'].max() - pd.Timedelta(days=TEST_DAYS)
train = df[df['date'] < test_start]
test = df[df['date'] >= test_start]
print(f"\ntrain: {len(train):,} rows | test: {len(test):,} rows "
      f"({test['date'].min().date()} -> {test['date'].max().date()})")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = WEATHER_FEATURES + LAG_FEATURES + ['street']
model = XGBRegressor(enable_categorical=True, random_state=42)
model.fit(train[FEATURES], train['pm25'])

test = test.copy()
test['pred'] = model.predict(test[FEATURES])
mse = mean_squared_error(test['pm25'], test['pred'])
r2 = r2_score(test['pm25'], test['pred'])
print(f"\nall-sensor model: MSE={mse:.2f}  RMSE={mse**0.5:.2f}  R2={r2:+.3f}")

print("\nper-sensor test performance:")
per_sensor = {}
for street in [s['street'] for s in SENSORS]:
    sub = test[test['street'] == street]
    if len(sub) == 0:
        continue
    m = mean_squared_error(sub['pm25'], sub['pred'])
    r = r2_score(sub['pm25'], sub['pred'])
    naive = mean_squared_error(sub['pm25'], sub['pm25_lag_1'])
    per_sensor[street] = (len(sub), m, r, naive)
    print(f"  {street:14} n={len(sub):3}  MSE={m:7.2f}  R2={r:+.3f}  "
          f"(persistence MSE={naive:7.2f})")

# --- 5. forecast the next days for every sensor ------------------
today = datetime.datetime.now() - datetime.timedelta(days=1)
forecast_weather = weather_fg.filter(weather_fg.date >= today).read()
forecast_weather['date'] = pd.to_datetime(
    forecast_weather['date']).dt.tz_localize(None)
forecast_weather = forecast_weather.sort_values('date').reset_index(drop=True)

# Forecasting several days ahead means the lag values are themselves
# predictions after the first step, so we roll them forward one day at a
# time rather than pretending we know them.
forecasts = []
for s in SENSORS:
    hist = all_aq[all_aq['street'] == s['street']].sort_values('date')
    recent = list(hist['pm25'].tail(3).astype(float))   # [t-3, t-2, t-1]
    rows = []
    for _, w in forecast_weather.iterrows():
        feat = {f: w[f] for f in WEATHER_FEATURES}
        feat['pm25_lag_1'] = recent[-1]
        feat['pm25_lag_2'] = recent[-2]
        feat['pm25_lag_3'] = recent[-3]
        feat['street'] = s['street']
        X = pd.DataFrame([feat])
        X['street'] = pd.Categorical(X['street'],
                                     categories=df['street'].cat.categories)
        p = float(model.predict(X[FEATURES])[0])
        rows.append({'date': w['date'], 'street': s['street'],
                     'city': s['city'], 'country': s['country'],
                     'predicted_pm25': p})
        recent.append(p)          # feed the prediction back in
        recent = recent[-3:]
    forecasts.append(pd.DataFrame(rows))
    print(f"  forecast for {s['street']:14} "
          f"{rows[0]['predicted_pm25']:.1f} -> {rows[-1]['predicted_pm25']:.1f}")

all_forecasts = pd.concat(forecasts, ignore_index=True)

# --- 6. dashboard: every sensor on one chart ---------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BANDS = [(0, 50, '#9cd84e', 'Good: 0-49'),
         (50, 100, '#facf39', 'Moderate: 50-99'),
         (100, 150, '#f99049', 'Unhealthy for Some: 100-149'),
         (150, 200, '#f65e5f', 'Unhealthy: 150-199')]
COLOURS = ['#c0392b', '#1e8449', '#6c3483']

os.makedirs('docs/air-quality/assets/img', exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 6))
top = max(60, float(all_forecasts['predicted_pm25'].max()) * 1.3)
for lo, hi, colour, _ in BANDS:
    if lo < top:
        ax.axhspan(lo, min(hi, top), color=colour, alpha=0.45, zorder=0)
for i, s in enumerate(SENSORS):
    sub = all_forecasts[all_forecasts['street'] == s['street']]
    ax.plot(sub['date'], sub['predicted_pm25'], color=COLOURS[i], linewidth=2,
            marker='o', markersize=6, label=s['street'], zorder=3)
ax.set_ylim(0, top)
ax.set_title(f'PM2.5 forecast for all {len(SENSORS)} air quality sensors '
             f'in {CITY}', fontsize=13)
ax.set_xlabel('Date'); ax.set_ylabel('PM2.5 (ug/m3)')
handles = [Patch(facecolor=c, alpha=0.45, label=l) for lo, _, c, l in BANDS
           if lo < top]
leg1 = ax.legend(handles=handles, title='Air Quality Categories',
                 loc='upper right', fontsize=8)
ax.add_artist(leg1)
ax.legend(loc='upper left', fontsize=9, title='Sensor')
ax.grid(alpha=0.25, zorder=1)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig('docs/air-quality/assets/img/pm25_forecast_all_sensors.png', dpi=120)
print("\nsaved docs/air-quality/assets/img/pm25_forecast_all_sensors.png")

# one panel per sensor, for the per-sensor view
fig2, axes = plt.subplots(len(SENSORS), 1, figsize=(11, 3.2 * len(SENSORS)),
                          sharex=True)
for i, s in enumerate(SENSORS):
    a = axes[i]
    sub = all_forecasts[all_forecasts['street'] == s['street']]
    for lo, hi, colour, _ in BANDS:
        if lo < top:
            a.axhspan(lo, min(hi, top), color=colour, alpha=0.4, zorder=0)
    a.plot(sub['date'], sub['predicted_pm25'], color=COLOURS[i], linewidth=2,
           marker='o', markersize=6, zorder=3)
    a.set_ylim(0, top)
    a.set_title(f"{s['street']}, {s['city']}", fontsize=11)
    a.set_ylabel('PM2.5')
    a.grid(alpha=0.25, zorder=1)
axes[-1].set_xlabel('Date')
fig2.autofmt_xdate(); fig2.tight_layout()
fig2.savefig('docs/air-quality/assets/img/pm25_forecast_per_sensor.png', dpi=120)
print("saved docs/air-quality/assets/img/pm25_forecast_per_sensor.png")

# --- 7. register the model ---------------------------------------
model_dir = "air_quality_model_all_sensors"
os.makedirs(model_dir + "/images", exist_ok=True)
model.save_model(model_dir + "/model.json")
fig.savefig(model_dir + "/images/forecast_all_sensors.png", dpi=120)

mr = project.get_model_registry()
metrics = {"MSE": str(mse), "R squared": str(r2),
           "n_sensors": str(len(SENSORS))}
for street, (n, m, r, naive) in per_sensor.items():
    metrics[f"MSE_{street}"] = str(m)
    metrics[f"R2_{street}"] = str(r)

aq_model = mr.python.create_model(
    name="air_quality_xgboost_all_sensors",
    metrics=metrics,
    feature_view=fv,
    description=f"PM2.5 predictor covering all {len(SENSORS)} sensors in {CITY}",
)
aq_model.save(model_dir)

print("=" * 60)
print("TASK 7 COMPLETE - all sensors in the city")
print("=" * 60)
print(f"  sensors  : {', '.join(s['street'] for s in SENSORS)}")
print(f"  rows     : {len(all_aq):,}")
print(f"  MSE      : {mse:.2f}   R2: {r2:+.3f}")
print(f"  model    : air_quality_xgboost_all_sensors v{aq_model.version}")
