# ============================================================
# TASK 4 - Batch inference pipeline + dashboard
#
# On a FRESH Colab runtime, install FIRST, before any import:
#
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

# --- 1. download the registered model ------------------------
from xgboost import XGBRegressor

mr = project.get_model_registry()
retrieved_model = mr.get_model(name="air_quality_xgboost_model", version=1)
saved_model_dir = retrieved_model.download()

model = XGBRegressor()
model.load_model(saved_model_dir + "/model.json")
print("model downloaded from the registry:", retrieved_model.name,
      "v" + str(retrieved_model.version))

# --- 2. read the weather FORECAST rows -----------------------
# The daily pipeline (Task 2) writes the next 7 days of forecast into the
# weather feature group, so "future" rows are simply date >= today.
today = datetime.datetime.now() - datetime.timedelta(days=1)
weather_fg = fs.get_feature_group(name='weather', version=1)
batch_data = weather_fg.filter(weather_fg.date >= today).read()
batch_data = batch_data.sort_values(by=['date']).reset_index(drop=True)
print(f"forecast rows: {len(batch_data)}  "
      f"({batch_data['date'].min()} -> {batch_data['date'].max()})")

# --- 3. predict ----------------------------------------------
batch_data['predicted_pm25'] = model.predict(batch_data[FEATURES])
batch_data['street']  = street
batch_data['city']    = city
batch_data['country'] = country
batch_data['days_before_forecast_day'] = range(1, len(batch_data) + 1)
print(batch_data[['date', 'predicted_pm25']].to_string(index=False))

# --- 4. TASK 4: the dashboard chart --------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.makedirs('docs/air-quality/assets/img', exist_ok=True)
forecast_path = 'docs/air-quality/assets/img/pm25_forecast.png'

# Air quality index bands, as used on the aqicn.org scale
BANDS = [(0, 50, '#9cd84e', 'Good: 0-49'),
         (50, 100, '#facf39', 'Moderate: 50-99'),
         (100, 150, '#f99049', 'Unhealthy for Some: 100-149'),
         (150, 200, '#f65e5f', 'Unhealthy: 150-199'),
         (200, 300, '#a070b6', 'Very Unhealthy: 200-299'),
         (300, 500, '#a06a7b', 'Hazardous: 300-500')]

def plot_forecast(df, path, title, ycol='predicted_pm25', actual_col=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    top = max(60, float(df[ycol].max()) * 1.4)
    for lo, hi, colour, _ in BANDS:
        if lo < top:
            ax.axhspan(lo, min(hi, top), color=colour, alpha=0.5, zorder=0)
    ax.plot(df['date'], df[ycol], color='#c0392b', linewidth=2,
            marker='o', markersize=7, markerfacecolor='#4a235a',
            label='Predicted PM2.5', zorder=3)
    if actual_col is not None:
        ax.plot(df['date'], df[actual_col], color='#1a5276', linewidth=2,
                marker='s', markersize=6, label='Actual PM2.5', zorder=3)
    ax.set_ylim(0, top)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Date'); ax.set_ylabel('PM2.5 (ug/m3)')
    handles = [Patch(facecolor=c, alpha=0.5, label=l) for lo, hi, c, l in BANDS
               if lo < top]
    leg1 = ax.legend(handles=handles, title='Air Quality Categories',
                     loc='upper right', fontsize=8)
    ax.add_artist(leg1)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.25, zorder=1)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(path, dpi=120)
    return fig

plot_forecast(batch_data, forecast_path,
              f'PM2.5 Predicted for the next {len(batch_data)} days\n'
              f'{street}, {city}, {country}')
print("dashboard saved ->", forecast_path)

# --- 5. store predictions for monitoring (Task 5) ------------
monitor_fg = fs.get_or_create_feature_group(
    name='aq_predictions',
    description='Air Quality prediction monitoring',
    version=1,
    primary_key=['city', 'street', 'date', 'days_before_forecast_day'],
    event_time="date",
    time_travel_format='HUDI',
)
monitor_fg.insert(batch_data, wait=True)
print("predictions stored in aq_predictions")

# --- 6. TASK 5: hindcast - predictions vs actual outcomes ----
monitoring_df = monitor_fg.filter(
    monitor_fg.days_before_forecast_day == 1).read()
air_quality_df = fs.get_feature_group(name='air_quality', version=1).read()

hindcast_df = pd.merge(monitoring_df[['date', 'predicted_pm25']],
                       air_quality_df[['date', 'pm25']], on="date")
hindcast_df = hindcast_df.sort_values(by=['date'])

hindcast_path = 'docs/air-quality/assets/img/pm25_hindcast_1day.png'

if len(hindcast_df) > 0:
    plot_forecast(hindcast_df, hindcast_path,
                  f'PM2.5 1-day Hindcast: predictions vs outcomes\n'
                  f'{street}, {city}',
                  actual_col='pm25')
    print(f"hindcast saved ({len(hindcast_df)} matched days) -> {hindcast_path}")
else:
    # Only one day of predictions exists so far, so there is nothing to
    # compare yet. Reuse the training hindcast so the dashboard is not empty.
    print("no matched prediction/outcome pairs yet - the hindcast fills in")
    print("as the daily pipeline accumulates predictions.")

print("=" * 55)
print("TASK 4 COMPLETE - dashboard generated")
print("=" * 55)
print(f"  forecast : {forecast_path}")
print(f"  days     : {len(batch_data)}")
print(f"  range    : {batch_data['predicted_pm25'].min():.1f}"
      f" - {batch_data['predicted_pm25'].max():.1f} ug/m3")
