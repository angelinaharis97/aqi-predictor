import requests
import os
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
DAYS_BACK = 30

# Work out our date range: 30 days ago until now
end_dt = datetime.utcnow()
start_dt = end_dt - timedelta(days=DAYS_BACK)
start_ts = int(start_dt.timestamp())
end_ts = int(end_dt.timestamp())

# --- Get 30 days of pollution history in ONE call ---
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={API_KEY}"
pollution_response = requests.get(pollution_url)
pollution_data = pollution_response.json()

# Build a lookup: {timestamp: pollution reading} for quick matching later
pollution_lookup = {entry['dt']: entry for entry in pollution_data['list']}

# --- Get 30 days of weather history in ONE call (from Open-Meteo) ---
start_date_str = start_dt.strftime('%Y-%m-%d')
end_date_str = end_dt.strftime('%Y-%m-%d')
weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LAT}&longitude={LON}&start_date={start_date_str}&end_date={end_date_str}&hourly=temperature_2m,relativehumidity_2m,windspeed_10m&timezone=UTC"
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

weather_times = weather_data['hourly']['time']
weather_temps = weather_data['hourly']['temperature_2m']
weather_humidity = weather_data['hourly']['relativehumidity_2m']
weather_wind = weather_data['hourly']['windspeed_10m']

# --- Build one feature row per day, using the noon (12:00) reading each day ---
feature_rows = []

for day_offset in range(DAYS_BACK):
    target_date = start_dt + timedelta(days=day_offset)
    target_dt = target_date.replace(hour=12, minute=0, second=0, microsecond=0)
    target_ts = int(target_dt.timestamp())
    target_time_str = target_dt.strftime("%Y-%m-%dT%H:00")

    # Find the matching pollution reading
    pollution_entry = pollution_lookup.get(target_ts)

    # Find the matching weather reading
    if target_time_str in weather_times:
        w_index = weather_times.index(target_time_str)
    else:
        w_index = None

    # Only build a row if we found both pieces of data for this day
    if pollution_entry and w_index is not None:
        feature_row = {
            "timestamp": target_dt.isoformat(),
            "hour": target_dt.hour,
            "day": target_dt.day,
            "month": target_dt.month,
            "day_of_week": target_dt.weekday(),
            "aqi": pollution_entry['main']['aqi'],
            "pm2_5": pollution_entry['components']['pm2_5'],
            "pm10": pollution_entry['components']['pm10'],
            "temp": weather_temps[w_index],
            "humidity": weather_humidity[w_index],
            "wind_speed": weather_wind[w_index]
        }
        feature_rows.append(feature_row)

print(f"Built {len(feature_rows)} feature rows out of {DAYS_BACK} days attempted")

# --- Send all rows to Supabase at once ---
if feature_rows:
    result = supabase.table("aqi_features").insert(feature_rows).execute()
    print("Backfill complete, saved to Supabase")
else:
    print("No rows were built — check the data above")
    