# ============================================================
# TASK 7 (fix) - direct multi-step forecasting for all sensors
#
# The recursive forecast compounds bias: only day 1 uses measured lag
# values, and every later day feeds the model's own prediction back in.
# Because the model runs slightly high, that error compounds - the 7-day
# curve ran to 102-114 ug/m3 against summer readings of 17-58.
#
# Direct multi-step forecasting avoids this. Train one model per horizon
# h = 1..7, each predicting pm25 at t+h from features measured at time t.
# No prediction is ever an input, so nothing compounds. Accuracy still
# decays with horizon - but it decays honestly instead of exploding.
#
# Upload the three CSVs first, then install:
#   !pip install -q hopsworks confluent-kafka deltalake xgboost \
#       scikit-learn matplotlib
# ============================================================
import getpass, os, json, datetime
import pandas as pd
import numpy as np

os.environ["HOPSWORKS_API_KEY"] = getpass.getpass("HOPSWORKS_API_KEY: ").strip()
os.environ["HOPSWORKS_PROJECT"] = input("HOPSWORKS_PROJECT: ").strip()
os.environ["HOPSWORKS_HOST"]    = "eu-west.cloud.hopsworks.ai"

import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
print("connected to", project.name)

SENSORS = ["Olivais", "Entrecampos", "Laranjeiro"]
CITY = "Lisboa"
LAG_FEATURES = ['pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3']
WEATHER_FEATURES = ['temperature_2m_mean', 'precipitation_sum',
                    'wind_speed_10m_max', 'wind_direction_10m_dominant']
FEATURES = WEATHER_FEATURES + LAG_FEATURES + ['street']
HORIZONS = list(range(1, 8))          # t+1 .. t+7
TEST_DAYS = 60

# --- 1. read the multi-sensor feature view -------------------
fv = fs.get_feature_view(name='air_quality_fv_all_sensors', version=1)
df = fv.query.read()
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
df = df.sort_values(['street', 'date']).reset_index(drop=True)
print(f"rows: {len(df):,}  sensors: {df['street'].nunique()}")

# --- 2. build one target column per horizon ------------------
# For horizon h, the target is pm25 h days AFTER the feature row. Built
# on a per-sensor continuous calendar so gaps never line up two rows
# that are not really h days apart.
def add_horizon_targets(g, street):
    g = g.set_index('date').sort_index()
    full = pd.date_range(g.index.min(), g.index.max(), freq='D')
    g = g.reindex(full)
    g.index.name = 'date'
    for h in HORIZONS:
        g[f'target_h{h}'] = g['pm25'].shift(-h)
    g = g.reset_index()
    # reindex() leaves NaN on gap rows, and groupby-apply would drop the
    # grouping column, so set it back explicitly
    g['street'] = street
    return g

df = pd.concat(
    [add_horizon_targets(g.drop(columns=['street']), street)
     for street, g in df.groupby('street', observed=True)],
    ignore_index=True)
df['street'] = df['street'].astype('category')
print(f"rows after building horizon targets: {len(df):,}")

test_start = df['date'].max() - pd.Timedelta(days=TEST_DAYS)

# --- 3. train one model per horizon --------------------------
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

models, metrics = {}, []
print(f"\n{'horizon':>8}  {'train':>7}  {'test':>5}  {'MSE':>8}  "
      f"{'RMSE':>6}  {'R2':>7}  {'persistence R2':>15}")

for h in HORIZONS:
    target = f'target_h{h}'
    sub = df.dropna(subset=FEATURES + [target] + ['pm25'])
    tr = sub[sub['date'] < test_start]
    te = sub[sub['date'] >= test_start]
    if len(te) == 0:
        continue

    m = XGBRegressor(enable_categorical=True, random_state=42)
    m.fit(tr[FEATURES], tr[target])
    pred = m.predict(te[FEATURES])

    mse = mean_squared_error(te[target], pred)
    r2 = r2_score(te[target], pred)
    # persistence at this horizon: predict t+h as the value at t
    p_r2 = r2_score(te[target], te['pm25'])
    p_mse = mean_squared_error(te[target], te['pm25'])

    models[h] = m
    metrics.append({'h': h, 'mse': mse, 'rmse': mse ** 0.5, 'r2': r2,
                    'persistence_mse': p_mse, 'persistence_r2': p_r2,
                    'n_train': len(tr), 'n_test': len(te)})
    print(f"{'t+' + str(h):>8}  {len(tr):>7,}  {len(te):>5}  {mse:>8.2f}  "
          f"{mse ** 0.5:>6.2f}  {r2:>+7.3f}  {p_r2:>+15.3f}")

met = pd.DataFrame(metrics)

# --- 4. forecast from the latest measured row ----------------
# Every horizon predicts from the SAME measured feature row, so no
# prediction is ever fed back in as an input.
latest = (df.dropna(subset=FEATURES + ['pm25'])
            .sort_values('date').groupby('street', observed=True).tail(1))
print(f"\nforecasting from measured features dated "
      f"{latest['date'].max().date()}")

rows = []
for _, base in latest.iterrows():
    for h in HORIZONS:
        if h not in models:
            continue
        X = pd.DataFrame([{f: base[f] for f in FEATURES}])
        X['street'] = pd.Categorical(X['street'],
                                     categories=df['street'].cat.categories)
        rows.append({
            'street': base['street'],
            'date': base['date'] + pd.Timedelta(days=h),
            'horizon': h,
            'predicted_pm25': float(models[h].predict(X[FEATURES])[0]),
        })
fc = pd.DataFrame(rows)

for s in SENSORS:
    v = fc[fc['street'] == s]['predicted_pm25'].tolist()
    print(f"  {s:14} " + "  ".join(f"{x:5.1f}" for x in v))

# --- 5. charts -----------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BANDS = [(0, 50, '#9cd84e', 'Good: 0-49'),
         (50, 100, '#facf39', 'Moderate: 50-99'),
         (100, 150, '#f99049', 'Unhealthy for Some: 100-149')]
COLOURS = {'Olivais': '#c0392b', 'Entrecampos': '#1e8449',
           'Laranjeiro': '#6c3483'}
os.makedirs('docs/air-quality/assets/img', exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 6))
top = max(60, float(fc['predicted_pm25'].max()) * 1.35)
for lo, hi, c, _ in BANDS:
    if lo < top:
        ax.axhspan(lo, min(hi, top), color=c, alpha=0.45, zorder=0)
for s in SENSORS:
    sub = fc[fc['street'] == s].sort_values('date')
    if len(sub):
        ax.plot(sub['date'], sub['predicted_pm25'], color=COLOURS[s],
                linewidth=2, marker='o', markersize=6, label=s, zorder=3)
ax.set_ylim(0, top)
ax.set_title(f'PM2.5 forecast for all {len(SENSORS)} sensors in {CITY}\n'
             f'direct multi-step: one model per horizon, no fed-back '
             f'predictions', fontsize=12)
ax.set_xlabel('Date'); ax.set_ylabel('PM2.5 (ug/m3)')
h1 = [Patch(facecolor=c, alpha=0.45, label=l) for lo, _, c, l in BANDS
      if lo < top]
leg1 = ax.legend(handles=h1, title='Air Quality Categories',
                 loc='upper right', fontsize=8)
ax.add_artist(leg1)
ax.legend(loc='upper left', fontsize=9, title='Sensor')
ax.grid(alpha=0.25, zorder=1)
fig.autofmt_xdate(); fig.tight_layout()
fig.savefig('docs/air-quality/assets/img/pm25_forecast_all_sensors.png', dpi=120)
print("\nsaved pm25_forecast_all_sensors.png")

# how skill decays with horizon - the honest picture
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(met['h'], met['r2'], color='#1a5276', linewidth=2, marker='o',
         markersize=7, label='Direct multi-step model')
ax2.plot(met['h'], met['persistence_r2'], color='#c0392b', linewidth=2,
         linestyle='--', marker='s', markersize=6,
         label='Persistence (predict t+h as today)')
ax2.axhline(0, color='#555', linewidth=1, linestyle=':')
ax2.set_xlabel('Forecast horizon (days ahead)')
ax2.set_ylabel('R² on the held-out test window')
ax2.set_title('Forecast skill decays with horizon\n'
              'R² > 0 beats predicting the mean; below 0 it does not',
              fontsize=12)
ax2.set_xticks(met['h'])
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig('docs/air-quality/assets/img/forecast_skill_by_horizon.png', dpi=120)
print("saved forecast_skill_by_horizon.png")

print("=" * 62)
print("DIRECT MULTI-STEP FORECAST COMPLETE")
print("=" * 62)
print(f"  models trained  : {len(models)} (one per horizon)")
print(f"  forecast range  : {fc['predicted_pm25'].min():.1f} - "
      f"{fc['predicted_pm25'].max():.1f} ug/m3")
print(f"  day-1 R2        : {met.iloc[0]['r2']:+.3f}")
print(f"  day-7 R2        : {met.iloc[-1]['r2']:+.3f}")
print("\n  compare with the recursive forecast, which ran to 102-114 ug/m3")
