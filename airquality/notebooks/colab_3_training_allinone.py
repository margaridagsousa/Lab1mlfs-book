# ============================================================
# TASK 3 - Training pipeline (+ Task 5 hindcast graph)
#
# On a FRESH Colab runtime, install FIRST, before any import:
#
#   !pip install -q hopsworks confluent-kafka deltalake xgboost \
#       scikit-learn matplotlib
#
# hsfs caches whether deltalake/confluent-kafka are present at import
# time, so installing them later does not take effect.
# ============================================================
import getpass, os, json
import pandas as pd

# --- 1. keys -------------------------------------------------
os.environ["HOPSWORKS_API_KEY"] = getpass.getpass("HOPSWORKS_API_KEY: ").strip()
os.environ["HOPSWORKS_PROJECT"] = input("HOPSWORKS_PROJECT: ").strip()
os.environ["HOPSWORKS_HOST"]    = "eu-west.cloud.hopsworks.ai"

import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
print("connected to", project.name)

# Sensor details were stored by the backfill (Task 1)
secrets = hopsworks.get_secrets_api()
location = json.loads(secrets.get_secret("SENSOR_LOCATION_JSON").value)
country, city, street = location['country'], location['city'], location['street']
print(f"sensor: {street}, {city}, {country}")

# --- 2. TASK 3 (1): select features into a Feature View ------
air_quality_fg = fs.get_feature_group(name='air_quality', version=1)
weather_fg     = fs.get_feature_group(name='weather', version=1)

# pm25 is the label; the weather columns are the features. Joined on
# city, aligned by date - this is what avoids training/serving skew.
selected_features = air_quality_fg.select(['pm25', 'date']).join(
    weather_fg.select_features(), on=['city'])

feature_view = fs.get_or_create_feature_view(
    name='air_quality_fv',
    description="weather features with air quality as the target",
    version=1,
    labels=['pm25'],
    query=selected_features,
)
print("feature view ready: air_quality_fv v1")

# --- 3. TASK 3 (2): read training data -----------------------
# Split by DATE, not randomly: predicting the future from the past means
# a random split would leak later days into training and flatter the model.
TEST_DAYS = int(os.getenv('TEST_DAYS', '60'))

df_all = feature_view.query.read()
df_all['date'] = pd.to_datetime(df_all['date'])
test_start = df_all['date'].max() - pd.Timedelta(days=TEST_DAYS)
print(f"rows: {len(df_all):,}  "
      f"({df_all['date'].min().date()} -> {df_all['date'].max().date()})")
print(f"test window starts: {test_start.date()}  (last {TEST_DAYS} days)")

X_train, X_test, y_train, y_test = feature_view.train_test_split(
    test_start=test_start)

# 'date' identifies the row; it is not a feature the model should learn from
X_features      = X_train.drop(columns=['date'])
X_test_features = X_test.drop(columns=['date'])
print(f"train: {len(X_train):,} rows | test: {len(X_test):,} rows")
print("features:", list(X_features.columns))

# --- 4. train the regressor ----------------------------------
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

xgb_regressor = XGBRegressor()
xgb_regressor.fit(X_features, y_train)

y_pred = xgb_regressor.predict(X_test_features)
mse = mean_squared_error(y_test.iloc[:, 0], y_pred)
r2  = r2_score(y_test.iloc[:, 0], y_pred)
print(f"\nMSE: {mse:.3f}")
print(f"R^2: {r2:.3f}")
print(f"RMSE: {mse ** 0.5:.3f} ug/m3  (typical error per day)")

# --- 5. TASK 5: hindcast graph (predictions vs outcomes) -----
df = y_test.copy()
df['predicted_pm25'] = y_pred
df['date'] = X_test['date']
df = df.sort_values(by=['date'])

model_dir = "air_quality_model"
images_dir = model_dir + "/images"
os.makedirs(images_dir, exist_ok=True)

import sys
sys.path.append('/content')          # if util.py was uploaded next to this
try:
    from airquality import util
    plt_obj = util.plot_air_quality_forecast(
        city, street, df, images_dir + "/pm25_hindcast.png", hindcast=True)
    print("hindcast plotted via util.plot_air_quality_forecast")
except Exception as e:
    # Standalone fallback so Task 5 does not depend on the repo being present
    print(f"(using built-in hindcast plot: {e})")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['date'], df['pm25'], label='Actual PM2.5',
            color='#2b6cb0', linewidth=1.8, marker='o', markersize=3)
    ax.plot(df['date'], df['predicted_pm25'], label='Predicted PM2.5',
            color='#e53e3e', linewidth=1.8, linestyle='--', marker='s', markersize=3)
    ax.set_title(f'PM2.5 Hindcast - predictions vs outcomes\n'
                 f'{street}, {city}  (MSE={mse:.2f}, R2={r2:.3f})')
    ax.set_xlabel('Date'); ax.set_ylabel('PM2.5 (ug/m3)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(images_dir + "/pm25_hindcast.png", dpi=120)
    print("hindcast saved ->", images_dir + "/pm25_hindcast.png")

# feature importance - useful to discuss at the defence
fig2, ax2 = plt.subplots(figsize=(8, 5))
plot_importance(xgb_regressor, ax=ax2)
fig2.tight_layout()
fig2.savefig(images_dir + "/feature_importance.png", dpi=120)
print("feature importance saved")

# --- 6. TASK 3 (3): register the model -----------------------
xgb_regressor.save_model(model_dir + "/model.json")

res_dict = {
    "MSE": str(mse),
    "R squared": str(r2),
    "test_days": str(TEST_DAYS),
    "train_samples": str(len(X_train)),
    "test_samples": str(len(X_test)),
}

mr = project.get_model_registry()
aq_model = mr.python.create_model(
    name="air_quality_xgboost_model",
    metrics=res_dict,
    feature_view=feature_view,
    description="Air Quality (PM2.5) predictor for " + street + ", " + city,
)
aq_model.save(model_dir)

print("=" * 55)
print("TASK 3 COMPLETE - model registered")
print("=" * 55)
print(f"  name    : air_quality_xgboost_model v{aq_model.version}")
print(f"  metrics : MSE={mse:.3f}  R2={r2:.3f}")
print(f"  trained : {len(X_train):,} rows | tested: {len(X_test):,} rows")
print(f"\nTASK 5: hindcast graph saved to {images_dir}/pm25_hindcast.png")
print("(display it below with the next cell)")
