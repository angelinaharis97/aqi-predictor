import os
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Step 1: Pull ALL rows (with pagination, since Supabase caps at 1000 per request) ---
print("Pulling all data from Supabase...")
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
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"Pulled {len(df)} rows")

# --- Step 2: Build a lookup so we can find "AQI at time X" quickly ---
aqi_by_time = df.set_index('timestamp')['aqi'].to_dict()

def find_future_aqi(row_time, hours_ahead):
    target_time = row_time + timedelta(hours=hours_ahead)
    # Round to nearest hour to match how our data is stored
    target_time = target_time.replace(minute=0, second=0, microsecond=0)
    return aqi_by_time.get(target_time)

# --- Step 3: For every row, look up its future AQI at 24h, 48h, 72h ahead ---
print("Building forecast targets...")
df['target_24h'] = df['timestamp'].apply(lambda t: find_future_aqi(t, 24))
df['target_48h'] = df['timestamp'].apply(lambda t: find_future_aqi(t, 48))
df['target_72h'] = df['timestamp'].apply(lambda t: find_future_aqi(t, 72))

rows_with_24h = df['target_24h'].notna().sum()
rows_with_48h = df['target_48h'].notna().sum()
rows_with_72h = df['target_72h'].notna().sum()
print(f"Rows with a valid 24h target: {rows_with_24h}")
print(f"Rows with a valid 48h target: {rows_with_48h}")
print(f"Rows with a valid 72h target: {rows_with_72h}")

# --- Step 4: Update each row in Supabase with its new target columns ---
import time

print("Updating Supabase with targets...")
update_count = 0

for _, row in df.iterrows():
    if pd.notna(row['target_24h']) or pd.notna(row['target_48h']) or pd.notna(row['target_72h']):
        retries = 3
        while retries > 0:
            try:
                supabase.table("aqi_features").update({
                    "target_24h": row['target_24h'] if pd.notna(row['target_24h']) else None,
                    "target_48h": row['target_48h'] if pd.notna(row['target_48h']) else None,
                    "target_72h": row['target_72h'] if pd.notna(row['target_72h']) else None,
                }).eq("id", row['id']).execute()
                break  # success, move to next row
            except Exception as e:
                retries -= 1
                if retries == 0:
                    print(f"  Failed on row id {row['id']} after 3 tries: {e}")
                else:
                    time.sleep(2)  # wait a moment before trying again

        update_count += 1
        if update_count % 500 == 0:
            print(f"  Updated {update_count} rows...")

print(f"Done! Updated {update_count} rows with forecast targets.")



