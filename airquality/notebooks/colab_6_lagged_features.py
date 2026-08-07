# ============================================================
# TASK 6 (Grade C) - lagged air quality features
#
# Adds pm25 from 1, 2 and 3 days ago as features, trains a second model,
# and compares it against the weather-only baseline.
#
# Rationale: PM2.5 is strongly autocorrelated - today's pollution
# resembles yesterday's - but the baseline model has no access to that.
# It only sees weather, which disperses pollution rather than causing it.
#
# Install first on a fresh runtime:
#   !pip install -q hopsworks confluent-kafka deltalake xgboost \
#       scikit-learn matplotlib
# ============================================================
import getpass, os, json
import pandas as pd
import numpy as np

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

WEATHER_FEATURES = ['temperature_2m_mean', 'precipitation_sum',
                    'wind_speed_10m_max', 'wind_direction_10m_dominant']
LAG_FEATURES = ['pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3']
TEST_DAYS = 60

# --- 1. build the lagged feature group ------------------------
aq_fg = fs.get_feature_group(name='air_quality', version=1)
aq_df = aq_fg.read()
aq_df['date'] = pd.to_datetime(aq_df['date']).dt.tz_localize(None)
aq_df = aq_df.sort_values('date').reset_index(drop=True)

# Lags must be computed on a continuous daily index. The sensor has gaps,
# and a naive shift() would silently pair a reading with one from weeks
# earlier, so reindex to every calendar day first and let the gaps be NaN.
full_range = pd.date_range(aq_df['date'].min(), aq_df['date'].max(), freq='D')
daily = aq_df.set_index('date').reindex(full_range)
daily.index.name = 'date'

for lag in (1, 2, 3):
    daily[f'pm25_lag_{lag}'] = daily['pm25'].shift(lag)

lagged = daily.reset_index()
lagged = lagged.dropna(subset=['pm25'] + LAG_FEATURES)
lagged['country'], lagged['city'], lagged['street'] = country, city, street
for c in ['pm25'] + LAG_FEATURES:
    lagged[c] = lagged[c].astype('float32')

print(f"rows with all 3 lags available: {len(lagged):,} "
      f"(from {len(aq_df):,} raw readings)")

# How strongly is pm25 autocorrelated? This is the whole premise.
print("\ncorrelation of pm25 with its own lags:")
for c in LAG_FEATURES:
    print(f"  {c}: {lagged['pm25'].corr(lagged[c]):+.3f}")

aq_lagged_fg = fs.get_or_create_feature_group(
    name='air_quality_lagged',
    description='Air quality with pm25 lagged 1, 2 and 3 days',
    version=1,
    primary_key=['country', 'city', 'street'],
    event_time='date',
    time_travel_format='HUDI',
)
aq_lagged_fg.insert(lagged[['date', 'pm25'] + LAG_FEATURES +
                           ['country', 'city', 'street']], wait=True)
print("air_quality_lagged feature group written")

# --- 2. feature view v2 (weather + lags) ---------------------
weather_fg = fs.get_feature_group(name='weather', version=1)

selected = aq_lagged_fg.select(['pm25', 'date'] + LAG_FEATURES).join(
    weather_fg.select_features(), on=['city'])

fv2 = fs.get_or_create_feature_view(
    name='air_quality_fv_lagged',
    description='weather + lagged pm25 features, air quality as target',
    version=1,
    labels=['pm25'],
    query=selected,
)
print("feature view ready: air_quality_fv_lagged v1")

# --- 3. train both models on the SAME rows -------------------
# A fair comparison requires identical train/test rows. Building the lags
# drops the first days and any row following a gap, so the baseline is
# re-trained here on this same subset rather than reusing Task 3's number.
df = fv2.query.read()
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
df = df.sort_values('date').reset_index(drop=True)

test_start = df['date'].max() - pd.Timedelta(days=TEST_DAYS)
train = df[df['date'] < test_start]
test = df[df['date'] >= test_start]
print(f"\ntrain: {len(train):,} rows | test: {len(test):,} rows")
print(f"test window: {test['date'].min().date()} -> {test['date'].max().date()}")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def fit_eval(features, label):
    m = XGBRegressor(random_state=42)
    m.fit(train[features], train['pm25'])
    pred = m.predict(test[features])
    mse = mean_squared_error(test['pm25'], pred)
    r2 = r2_score(test['pm25'], pred)
    print(f"{label:28} MSE={mse:8.2f}  RMSE={mse**0.5:6.2f}  R2={r2:+.3f}")
    return m, pred, mse, r2

print()
m1, pred1, mse1, r21 = fit_eval(WEATHER_FEATURES, "v1 weather only")
m2, pred2, mse2, r22 = fit_eval(WEATHER_FEATURES + LAG_FEATURES,
                                "v2 weather + lags")

# A naive "tomorrow equals today" baseline. If the model cannot beat
# this, the lag features are doing the work and the model is not.
naive_mse = mean_squared_error(test['pm25'], test['pm25_lag_1'])
naive_r2 = r2_score(test['pm25'], test['pm25_lag_1'])
print(f"{'persistence (pm25_lag_1)':28} MSE={naive_mse:8.2f}  "
      f"RMSE={naive_mse**0.5:6.2f}  R2={naive_r2:+.3f}")

improvement = (mse1 - mse2) / mse1 * 100
print(f"\nMSE improvement from lag features: {improvement:+.1f}%")

# --- 4. register model v2 ------------------------------------
model_dir = "air_quality_model_lagged"
images_dir = model_dir + "/images"
os.makedirs(images_dir, exist_ok=True)
m2.save_model(model_dir + "/model.json")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import plot_importance

# comparison chart for the report
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(test['date'], test['pm25'], color='#1a5276', linewidth=2,
        marker='s', markersize=4, label='Actual PM2.5')
ax.plot(test['date'], pred1, color='#c0392b', linewidth=1.6, linestyle='--',
        marker='o', markersize=3, alpha=0.8,
        label=f'v1 weather only (R2={r21:+.3f})')
ax.plot(test['date'], pred2, color='#1e8449', linewidth=1.8, linestyle='-.',
        marker='^', markersize=3,
        label=f'v2 weather + lags (R2={r22:+.3f})')
ax.set_title(f'Effect of lagged PM2.5 features - {street}, {city}\n'
             f'MSE {mse1:.1f} -> {mse2:.1f} ({improvement:+.1f}%)', fontsize=12)
ax.set_xlabel('Date'); ax.set_ylabel('PM2.5 (ug/m3)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(images_dir + '/lag_comparison.png', dpi=120)

os.makedirs('docs/air-quality/assets/img', exist_ok=True)
fig.savefig('docs/air-quality/assets/img/lag_comparison.png', dpi=120)

fig2, ax2 = plt.subplots(figsize=(8, 5))
plot_importance(m2, ax=ax2)
fig2.tight_layout()
fig2.savefig(images_dir + '/feature_importance.png', dpi=120)

mr = project.get_model_registry()
aq_model2 = mr.python.create_model(
    name="air_quality_xgboost_model_lagged",
    metrics={"MSE": str(mse2), "R squared": str(r22),
             "MSE_baseline": str(mse1), "R2_baseline": str(r21),
             "improvement_pct": str(improvement)},
    feature_view=fv2,
    description=f"PM2.5 predictor with 1/2/3-day lags for {street}, {city}",
)
aq_model2.save(model_dir)

print("=" * 60)
print("TASK 6 COMPLETE")
print("=" * 60)
print(f"  {'model':28} {'MSE':>9} {'RMSE':>7} {'R2':>8}")
print(f"  {'v1 weather only':28} {mse1:9.2f} {mse1**0.5:7.2f} {r21:+8.3f}")
print(f"  {'v2 weather + lags':28} {mse2:9.2f} {mse2**0.5:7.2f} {r22:+8.3f}")
print(f"  {'persistence baseline':28} {naive_mse:9.2f} "
      f"{naive_mse**0.5:7.2f} {naive_r2:+8.3f}")
print(f"\n  MSE improvement: {improvement:+.1f}%")
print(f"  registered as air_quality_xgboost_model_lagged "
      f"v{aq_model2.version}")
