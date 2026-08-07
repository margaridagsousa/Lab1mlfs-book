# ============================================================
# TASK 1 - run this ONE cell after restarting the runtime
#
# On a FRESH runtime, install everything BEFORE the first hopsworks
# import - hsfs caches whether deltalake/confluent-kafka are present at
# import time, and installing them later does not update those flags:
#
#   !pip install -q hopsworks confluent-kafka deltalake \
#       openmeteo-requests requests-cache retry-requests geopy xgboost
# ============================================================
import getpass, os, json, datetime
import pandas as pd

# --- 1. keys -------------------------------------------------
os.environ["HOPSWORKS_API_KEY"] = getpass.getpass("HOPSWORKS_API_KEY: ").strip()
os.environ["HOPSWORKS_PROJECT"] = input("HOPSWORKS_PROJECT (project name): ").strip()
os.environ["HOPSWORKS_HOST"]    = "eu-west.cloud.hopsworks.ai"
AQICN_API_KEY = getpass.getpass("AQICN_API_KEY: ").strip()

# --- 2. sensor ----------------------------------------------
country, city, street = "Portugal", "Lisboa", "Olivais"
aqicn_url = "https://api.waqi.info/feed/@10513"
latitude, longitude = 38.76888900025, -9.108055999671
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)

# --- 3. air quality from the uploaded CSV --------------------
csv_file = "olivais-lisboa.csv"          # re-upload if the runtime was reset
df = pd.read_csv(csv_file, parse_dates=['date'], skipinitialspace=True)
df_aq = df[['date', 'pm25']].copy()
df_aq['pm25'] = pd.to_numeric(df_aq['pm25'], errors='coerce').astype('float32')
df_aq = df_aq.dropna()
df_aq['country'], df_aq['city'] = country, city
df_aq['street'],  df_aq['url']  = street, aqicn_url
print(f"air quality rows: {len(df_aq)}  "
      f"({df_aq['date'].min().date()} -> {df_aq['date'].max().date()})")

# --- 4. historical weather from Open-Meteo -------------------
import openmeteo_requests, requests_cache
from retry_requests import retry

openmeteo = openmeteo_requests.Client(
    session=retry(requests_cache.CachedSession('.cache', expire_after=-1),
                  retries=5, backoff_factor=0.2))
resp = openmeteo.weather_api(
    "https://archive-api.open-meteo.com/v1/archive",
    params={"latitude": latitude, "longitude": longitude,
            "start_date": df_aq['date'].min().strftime('%Y-%m-%d'),
            "end_date": str(yesterday),
            "daily": ["temperature_2m_mean", "precipitation_sum",
                      "wind_speed_10m_max", "wind_direction_10m_dominant"]})[0]
daily = resp.Daily()
weather_df = pd.DataFrame({
    "date": pd.date_range(pd.to_datetime(daily.Time(), unit="s"),
                          pd.to_datetime(daily.TimeEnd(), unit="s"),
                          freq=pd.Timedelta(seconds=daily.Interval()),
                          inclusive="left"),
    "temperature_2m_mean":         daily.Variables(0).ValuesAsNumpy(),
    "precipitation_sum":           daily.Variables(1).ValuesAsNumpy(),
    "wind_speed_10m_max":          daily.Variables(2).ValuesAsNumpy(),
    "wind_direction_10m_dominant": daily.Variables(3).ValuesAsNumpy(),
}).dropna()
weather_df['city'] = city
print(f"weather rows: {len(weather_df)}")

# --- 5. connect ----------------------------------------------
import deltalake, confluent_kafka, hopsworks
print("deltalake", deltalake.__version__, "| confluent_kafka", confluent_kafka.__version__)

project = hopsworks.login()
fs = project.get_feature_store()
print("connected to", project.name)

# --- 6. secrets for the daily pipeline (Task 2) --------------
secrets = hopsworks.get_secrets_api()
def put_secret(name, value):
    try:
        s = secrets.get_secret(name)
        if s is not None:
            s.delete()
    except Exception:
        pass
    secrets.create_secret(name, value)
    print("stored", name)

put_secret("AQICN_API_KEY", AQICN_API_KEY)
put_secret("SENSOR_LOCATION_JSON", json.dumps({
    "country": country, "city": city, "street": street,
    "aqicn_url": aqicn_url, "latitude": latitude, "longitude": longitude}))

# --- 7. FEATURE GROUP 1/2: air_quality -----------------------
air_quality_fg = fs.get_or_create_feature_group(
    name='air_quality',
    description='Air Quality characteristics of each day',
    version=1,
    primary_key=['country', 'city', 'street'],
    event_time='date',
    time_travel_format='HUDI',
)
air_quality_fg.insert(df_aq, wait=True)
print("air_quality written")

# --- 8. FEATURE GROUP 2/2: weather ---------------------------
weather_fg = fs.get_or_create_feature_group(
    name='weather',
    description='Weather characteristics of each day',
    version=1,
    primary_key=['city'],
    event_time='date',
    time_travel_format='HUDI',
)
weather_fg.insert(weather_df, wait=True)
print("weather written")

# --- 9. verify -----------------------------------------------
print("=" * 55)
print("TASK 1 COMPLETE - 2 Feature Groups registered")
print("=" * 55)
for name in ("air_quality", "weather"):
    fg = fs.get_feature_group(name=name, version=1)
    print(f"{name} (v{fg.version}): {len(fg.read()):,} rows")
    print(f"   features: {[f.name for f in fg.features]}")
