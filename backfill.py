import requests
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LAT = 24.8607
LON = 67.0011
DAYS_BACK = 365

end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=DAYS_BACK)

# --- Step 1: Clear out the old small dataset for a clean, consistent one ---
print("Clearing old data...")
supabase.table("aqi_features").delete().neq("id", 0).execute()

# --- Step 2: Get pollution history in 30-day chunks (13 chunks for a year) ---
print("Fetching pollution history in chunks...")
pollution_lookup = {}
chunk_start = start_dt

while chunk_start < end_dt:
    chunk_end = min(chunk_start + timedelta(days=30), end_dt)
    start_ts = int(chunk_start.timestamp())
    end_ts = int(chunk_end.timestamp())

    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if 'list' in data:
        for entry in data['list']:
            pollution_lookup[entry['dt']] = entry
        print(f"  Chunk {chunk_start.date()} to {chunk_end.date()}: {len(data['list'])} readings")
    else:
        print(f"  Chunk {chunk_start.date()} to {chunk_end.date()}: no data returned")

    chunk_start = chunk_end
    time.sleep(1)  # small pause to be polite to the API

print(f"Total pollution readings collected: {len(pollution_lookup)}")

# --- Step 3: Get weather history for the whole year in one call ---
print("Fetching weather history...")
start_date_str = start_dt.strftime('%Y-%m-%d')
end_date_str = end_dt.strftime('%Y-%m-%d')
weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date_str}&end_date={end_date_str}&hourly=temperature_2m,relativehumidity_2m,windspeed_10m&timezone=UTC"
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

weather_times = weather_data['hourly']['time']
weather_temps = weather_data['hourly']['temperature_2m']
weather_humidity = weather_data['hourly']['relativehumidity_2m']
weather_wind = weather_data['hourly']['windspeed_10m']

print(f"Total weather readings collected: {len(weather_times)}")

# --- Step 4: Match pollution + weather by timestamp, build feature rows ---
print("Matching and building feature rows...")
feature_rows = []

for i, time_str in enumerate(weather_times):
    dt = datetime.fromisoformat(time_str)
    ts = int(dt.timestamp())

    pollution_entry = pollution_lookup.get(ts)

    if pollution_entry and weather_temps[i] is not None:
        feature_row = {
            "timestamp": dt.isoformat(),
            "hour": dt.hour,
            "day": dt.day,
            "month": dt.month,
            "day_of_week": dt.weekday(),
            "aqi": pollution_entry['main']['aqi'],
            "pm2_5": pollution_entry['components']['pm2_5'],
            "pm10": pollution_entry['components']['pm10'],
            "o3": pollution_entry['components']['o3'],
            "no2": pollution_entry['components']['no2'],
            "so2": pollution_entry['components']['so2'],
            "co": pollution_entry['components']['co'],
            "temp": weather_temps[i],
            "humidity": weather_humidity[i],
            "wind_speed": weather_wind[i]
        }
        feature_rows.append(feature_row)

print(f"Built {len(feature_rows)} matched feature rows")

# --- Step 5: Insert in batches (large datasets need chunked inserts) ---
print("Uploading to Supabase in batches...")
batch_size = 500
for i in range(0, len(feature_rows), batch_size):
    batch = feature_rows[i:i + batch_size]
    supabase.table("aqi_features").insert(batch).execute()
    print(f"  Uploaded rows {i} to {i + len(batch)}")

print("Backfill complete!")

