import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Pull all data (paginated) ---
print("Pulling data from Supabase...")
all_data = []
batch_size = 1000
offset = 0
while True:
    response = supabase.table("aqi_features").select("*").range(offset, offset + batch_size - 1).execute()
    if not response.data:
        break
    all_data.extend(response.data)
    offset += batch_size

df = pd.DataFrame(all_data)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
print(f"Pulled {len(df)} rows")

os.makedirs("eda_plots", exist_ok=True)

# --- Chart 1: Correlation heatmap ---
print("Building correlation heatmap...")
cols = ["aqi", "pm2_5", "pm10", "o3", "no2", "so2", "co", "temp", "humidity", "wind_speed"]
corr = df[cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn_r", center=0, ax=ax)
ax.set_title("Feature Correlation with AQI — Karachi")
fig.savefig("eda_plots/correlation_heatmap.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved eda_plots/correlation_heatmap.png")

# --- Chart 2: Average AQI by hour of day ---
print("Building hourly pattern chart...")
hourly_avg = df.groupby("hour")["aqi"].mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(hourly_avg.index, hourly_avg.values, color="#3182ce")
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Average AQI")
ax.set_title("Average AQI by Hour of Day — Karachi")
ax.set_xticks(range(24))
fig.savefig("eda_plots/hourly_pattern.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved eda_plots/hourly_pattern.png")

# --- Print key findings ---
print("\n=== EDA SUMMARY ===")
print(f"Total records: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nStrongest correlations with AQI:")
aqi_corr = corr["aqi"].drop("aqi").sort_values(key=abs, ascending=False)
for feature, val in aqi_corr.items():
    print(f"  {feature}: r = {val:.3f}")
print(f"\nPeak pollution hour: {hourly_avg.idxmax()}:00 (avg AQI {hourly_avg.max():.2f})")
print(f"Cleanest hour: {hourly_avg.idxmin()}:00 (avg AQI {hourly_avg.min():.2f})")
