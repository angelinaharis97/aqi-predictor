import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

# Call 1: Air pollution data (what you already had)
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
pollution_response = requests.get(pollution_url)
pollution_data = pollution_response.json()

# Call 2: Weather data (new)
weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

print("POLLUTION DATA:")
print(pollution_data)

print("\nWEATHER DATA:")
print(weather_data)