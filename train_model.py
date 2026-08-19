import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from huggingface_hub import HfApi

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Pull ALL rows (paginated, since Supabase caps at 1000 per request) ---
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
print(f"Pulled {len(df)} rows")

feature_columns = ["hour", "day", "month", "day_of_week", "pm2_5", "pm10", "o3", "no2", "so2", "co", "temp", "humidity", "wind_speed"]

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id="angelinaharis/aqi-predictor-model", repo_type="model", exist_ok=True)

# --- Train one model per forecast horizon ---
horizons = {
    "24h": "target_24h",
    "48h": "target_48h",
    "72h": "target_72h"
}

results = {}

for horizon_name, target_col in horizons.items():
    print(f"\n=== Training for {horizon_name} horizon ===")

    # Only use rows where we actually know the real future AQI
    valid_rows = df.dropna(subset=feature_columns + [target_col])
    print(f"Valid rows for {horizon_name}: {len(valid_rows)}")

    X = valid_rows[feature_columns]
    y = valid_rows[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_r2 = -999
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        print(f"  {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    print(f"  Best for {horizon_name}: {best_name} (R2={best_r2:.3f})")
    results[horizon_name] = {"model": best_name, "r2": best_r2}

    # Save and upload this horizon's best model
    filename = f"aqi_model_{horizon_name}.joblib"
    joblib.dump(best_model, filename)
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id="angelinaharis/aqi-predictor-model",
        repo_type="model"
    )
    print(f"  Uploaded {filename} to Hugging Face")

print("\n=== SUMMARY ===")
for horizon, info in results.items():
    print(f"{horizon}: best model = {info['model']}, R2 = {info['r2']:.3f}")
