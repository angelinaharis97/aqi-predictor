import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

# Call 1: Air pollution data
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
pollution_response = requests.get(pollution_url)
pollution_data = pollution_response.json()

# Call 2: Weather data
weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

# Pull out the specific numbers we care about
pollution_info = pollution_data['list'][0]
aqi = pollution_info['main']['aqi']
pm2_5 = pollution_info['components']['pm2_5']
pm10 = pollution_info['components']['pm10']

temp = weather_data['main']['temp']
humidity = weather_data['main']['humidity']
wind_speed = weather_data['wind']['speed']

# Time-based features
now = datetime.now()
hour = now.hour
day = now.day
month = now.month
day_of_week = now.weekday()  # Monday = 0, Sunday = 6

# Build one clean "row" of features
feature_row = {
    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    "hour": hour,
    "day": day,
    "month": month,
    "day_of_week": day_of_week,
    "aqi": aqi,
    "pm2_5": pm2_5,
    "pm10": pm10,
    "temp": temp,
    "humidity": humidity,
    "wind_speed": wind_speed
}

print("FEATURE ROW:")
print(feature_row)
