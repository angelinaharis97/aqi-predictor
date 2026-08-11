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
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="angelinaharis/aqi-predictor-model", filename="aqi_model.joblib")
model = joblib.load(model_path)

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
    import pandas as pd
import shap
import matplotlib.pyplot as plt

# --- History Chart ---
st.subheader("Recent AQI History")

history_response = supabase.table("aqi_features").select("timestamp", "aqi").order("timestamp").execute()
history_df = pd.DataFrame(history_response.data)
history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], format='mixed')
history_df = history_df.set_index('timestamp')

st.line_chart(history_df['aqi'])

# --- Hazard Alert ---
if current_aqi >= 4:
    st.error(f"⚠️ Hazardous air quality alert! Current AQI is {current_aqi} ({aqi_labels[current_aqi]}). Consider limiting outdoor activity.")
else:
    st.success(f"Air quality is currently at an acceptable level ({aqi_labels[current_aqi]}).")

# --- SHAP Explanation ---
st.subheader("Why this prediction? (Feature Importance)")

feature_columns = ["hour", "day", "month", "day_of_week", "pm2_5", "pm10", "temp", "humidity", "wind_speed"]

# Use recent historical data as the "background" for SHAP to compare against
background_data = pd.DataFrame(supabase.table("aqi_features").select("*").execute().data)[feature_columns]

explainer = shap.Explainer(model, background_data)
sample_input = pd.DataFrame([[
    datetime.now().hour, datetime.now().day, datetime.now().month, datetime.now().weekday(),
    current_pm25, current_pm10, current_temp, current_humidity, current_wind
]], columns=feature_columns)

shap_values = explainer(sample_input)

fig, ax = plt.subplots()
shap.plots.bar(shap_values[0], show=False)
st.pyplot(fig)


