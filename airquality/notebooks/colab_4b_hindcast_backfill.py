# ============================================================
# Task 5 - backfill the monitoring feature group with historical
# predictions so the hindcast chart has real history in it.
#
# The daily pipeline only starts accumulating predictions from the day
# it first runs, so a fresh deployment has a single matched day. Here we
# replay the model over past weather that is already in the feature
# store and pair each prediction with the measured outcome.
#
# These are honest out-of-sample predictions for the test window: the
# model was trained on data up to 2026-06-07, so anything after that was
# never seen during training.
#
# Install first on a fresh runtime:
#   !pip install -q hopsworks confluent-kafka deltalake xgboost matplotlib
# ============================================================
import getpass, os, json, datetime
import pandas as pd

os.environ["HOPSWORKS_API_KEY"] = getpass.getpass("HOPSWORKS_API_KEY: ").strip()
os.environ["HOPSWORKS_PROJECT"] = input("HOPSWORKS_PROJECT: ").strip()
os.environ["HOPSWORKS_HOST"]    = "eu-west.cloud.hopsworks.ai"

import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

secrets = hopsworks.get_secrets_api()
location = json.loads(secrets.get_secret("SENSOR_LOCATION_JSON").value)
country, city, street = location['country'], location['city'], location['street']
print(f"connected to {project.name} | sensor: {street}, {city}")

FEATURES = ['temperature_2m_mean', 'precipitation_sum',
            'wind_speed_10m_max', 'wind_direction_10m_dominant']

# How far back to replay. The model's test window was the last 60 days,
# so 60 keeps every backfilled prediction genuinely out-of-sample.
HINDCAST_DAYS = int(os.getenv('HINDCAST_DAYS', '60'))

# --- 1. model + history --------------------------------------
from xgboost import XGBRegressor

mr = project.get_model_registry()
retrieved_model = mr.get_model(name="air_quality_xgboost_model", version=1)
model = XGBRegressor()
model.load_model(retrieved_model.download() + "/model.json")

weather_df = fs.get_feature_group(name='weather', version=1).read()
aq_df = fs.get_feature_group(name='air_quality', version=1).read()

for d in (weather_df, aq_df):
    d['date'] = pd.to_datetime(d['date']).dt.tz_localize(None)

# --- 2. replay the model over past weather -------------------
cutoff = aq_df['date'].max() - pd.Timedelta(days=HINDCAST_DAYS)
past = weather_df[(weather_df['date'] >= cutoff) &
                  (weather_df['date'] <= aq_df['date'].max())].copy()
past = past.sort_values('date').reset_index(drop=True)

past['predicted_pm25'] = model.predict(past[FEATURES])
past['street'], past['city'], past['country'] = street, city, country
# These stand for forecasts made one day ahead, which is what the
# 1-day hindcast chart selects on.
past['days_before_forecast_day'] = 1

print(f"replayed {len(past)} days "
      f"({past['date'].min().date()} -> {past['date'].max().date()})")

# --- 3. write them into the monitoring feature group ---------
monitor_fg = fs.get_or_create_feature_group(
    name='aq_predictions',
    description='Air Quality prediction monitoring',
    version=1,
    primary_key=['city', 'street', 'date', 'days_before_forecast_day'],
    event_time="date",
    time_travel_format='HUDI',
)
monitor_fg.insert(past, wait=True)
print("backfilled predictions written to aq_predictions")

# --- 4. rebuild the hindcast chart ---------------------------
hindcast_df = pd.merge(past[['date', 'predicted_pm25']],
                       aq_df[['date', 'pm25']], on='date').sort_values('date')
print(f"matched prediction/outcome pairs: {len(hindcast_df)}")

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(hindcast_df['pm25'], hindcast_df['predicted_pm25'])
r2 = r2_score(hindcast_df['pm25'], hindcast_df['predicted_pm25'])
bias = (hindcast_df['predicted_pm25'] - hindcast_df['pm25']).mean()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BANDS = [(0, 50, '#9cd84e', 'Good: 0-49'),
         (50, 100, '#facf39', 'Moderate: 50-99'),
         (100, 150, '#f99049', 'Unhealthy for Some: 100-149'),
         (150, 200, '#f65e5f', 'Unhealthy: 150-199')]

os.makedirs('docs/air-quality/assets/img', exist_ok=True)
path = 'docs/air-quality/assets/img/pm25_hindcast_1day.png'

fig, ax = plt.subplots(figsize=(13, 6))
top = max(60, float(max(hindcast_df['predicted_pm25'].max(),
                        hindcast_df['pm25'].max())) * 1.25)
for lo, hi, colour, _ in BANDS:
    if lo < top:
        ax.axhspan(lo, min(hi, top), color=colour, alpha=0.45, zorder=0)

ax.plot(hindcast_df['date'], hindcast_df['pm25'], color='#1a5276',
        linewidth=2, marker='s', markersize=5, label='Actual PM2.5', zorder=3)
ax.plot(hindcast_df['date'], hindcast_df['predicted_pm25'], color='#c0392b',
        linewidth=2, linestyle='--', marker='o', markersize=5,
        label='Predicted PM2.5', zorder=3)

ax.set_ylim(0, top)
ax.set_xlim(hindcast_df['date'].min(), hindcast_df['date'].max())
ax.set_title(f'PM2.5 1-day Hindcast: predictions vs outcomes\n'
             f'{street}, {city}  -  {len(hindcast_df)} days  '
             f'(MSE={mse:.1f}, R2={r2:.3f}, mean bias={bias:+.1f} ug/m3)',
             fontsize=12)
ax.set_xlabel('Date'); ax.set_ylabel('PM2.5 (ug/m3)')
handles = [Patch(facecolor=c, alpha=0.45, label=l) for lo, _, c, l in BANDS
           if lo < top]
leg1 = ax.legend(handles=handles, title='Air Quality Categories',
                 loc='upper right', fontsize=8)
ax.add_artist(leg1)
ax.legend(loc='upper left', fontsize=9)
ax.grid(alpha=0.25, zorder=1)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(path, dpi=120)

print("=" * 55)
print("HINDCAST BACKFILLED")
print("=" * 55)
print(f"  days      : {len(hindcast_df)}")
print(f"  MSE       : {mse:.2f}")
print(f"  R2        : {r2:.3f}")
print(f"  mean bias : {bias:+.2f} ug/m3 "
      f"({'over' if bias > 0 else 'under'}-predicting)")
print(f"  saved     : {path}")
