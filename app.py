import streamlit as st
import os
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

LAT = 24.8607
LON = 67.0011

st.title("🌫️ Karachi AQI Predictor")
st.write("Real-time air quality monitoring and 3-day forecast")

# Get current pollution reading
pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
pollution_response = requests.get(pollution_url)
pollution_data = pollution_response.json()
current_aqi = pollution_data['list'][0]['main']['aqi']

aqi_labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}

st.metric("Current AQI", f"{current_aqi} ({aqi_labels[current_aqi]})")
import joblib
from datetime import datetime, timedelta

# Get current weather too (needed for the forecast inputs)
weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

current_temp = weather_data['main']['temp']
current_humidity = weather_data['main']['humidity']
current_wind = weather_data['wind']['speed']
current_pm25 = pollution_data['list'][0]['components']['pm2_5']
current_pm10 = pollution_data['list'][0]['components']['pm10']

# Load your trained model
model = joblib.load("aqi_model.joblib")

st.subheader("3-Day Forecast")

for i in range(1, 4):
    future_date = datetime.now() + timedelta(days=i)
    
    # Build the input row the same way we built training data
    input_data = [[
        future_date.hour,
        future_date.day,
        future_date.month,
        future_date.weekday(),
        current_pm25,
        current_pm10,
        current_temp,
        current_humidity,
        current_wind
    ]]
    
    predicted_aqi = model.predict(input_data)[0]
    predicted_aqi_rounded = round(predicted_aqi)
    
    st.write(f"**{future_date.strftime('%A, %b %d')}**: Predicted AQI ≈ {predicted_aqi:.1f} ({aqi_labels.get(predicted_aqi_rounded, 'Unknown')})")
    