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
from huggingface_hub import hf_hub_download

# Get current weather too (needed for the forecast inputs)
weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

current_temp = weather_data['main']['temp']
current_humidity = weather_data['main']['humidity']
current_wind = weather_data['wind']['speed']
current_pollutants = pollution_data['list'][0]['components']
current_pm25 = current_pollutants['pm2_5']
current_pm10 = current_pollutants['pm10']
current_o3 = current_pollutants['o3']
current_no2 = current_pollutants['no2']
current_so2 = current_pollutants['so2']
current_co = current_pollutants['co']

now = datetime.now()

# Load the three specialist models, one per horizon
model_24h_path = hf_hub_download(repo_id="angelinaharis/aqi-predictor-model", filename="aqi_model_24h.joblib")
model_48h_path = hf_hub_download(repo_id="angelinaharis/aqi-predictor-model", filename="aqi_model_48h.joblib")
model_72h_path = hf_hub_download(repo_id="angelinaharis/aqi-predictor-model", filename="aqi_model_72h.joblib")

model_24h = joblib.load(model_24h_path)
model_48h = joblib.load(model_48h_path)
model_72h = joblib.load(model_72h_path)

# Today's conditions are the input for ALL horizons -- each model
# has separately learned what THIS input means for its own future point
input_data = [[
    now.hour, now.day, now.month, now.weekday(),
    current_pm25, current_pm10, current_o3, current_no2, current_so2, current_co,
    current_temp, current_humidity, current_wind
]]

st.subheader("3-Day Forecast")

horizon_models = {
    1: ("24 hours", model_24h),
    2: ("48 hours", model_48h),
    3: ("72 hours", model_72h)
}

for days_ahead, (label, model) in horizon_models.items():
    future_date = now + timedelta(days=days_ahead)
    predicted_aqi = model.predict(input_data)[0]
    predicted_aqi_rounded = round(predicted_aqi)
    st.write(f"**{future_date.strftime('%A, %b %d')}** ({label} ahead): Predicted AQI ≈ {predicted_aqi:.1f} ({aqi_labels.get(predicted_aqi_rounded, 'Unknown')})")

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

# --- SHAP Explanation (for the 24h forecast model) ---
st.subheader("Why this 24h prediction? (Feature Importance)")

feature_columns = ["hour", "day", "month", "day_of_week", "pm2_5", "pm10", "o3", "no2", "so2", "co", "temp", "humidity", "wind_speed"]

# Use recent historical data as the "background" for SHAP to compare against
background_response = supabase.table("aqi_features").select("*").limit(500).execute()
background_data = pd.DataFrame(background_response.data)[feature_columns]

explainer = shap.Explainer(model_24h, background_data)
sample_input = pd.DataFrame(input_data, columns=feature_columns)

shap_values = explainer(sample_input)

fig, ax = plt.subplots()
shap.plots.bar(shap_values[0], show=False)
st.pyplot(fig)

