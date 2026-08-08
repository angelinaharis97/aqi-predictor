import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Pull ALL rows from your feature table
response = supabase.table("aqi_features").select("*").execute()
data = response.data

# Turn it into a proper table (like an Excel sheet inside Python)
df = pd.DataFrame(data)

print(f"Pulled {len(df)} rows")
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# These are the columns we'll use to PREDICT with (the inputs)
feature_columns = ["hour", "day", "month", "day_of_week", "pm2_5", "pm10", "temp", "humidity", "wind_speed"]

# This is what we're trying to predict (the answer)
target_column = "aqi"

X = df[feature_columns]  # inputs
y = df[target_column]    # correct answers

# Split data: most of it for teaching the model, some held back to test it honestly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test it on the data it HASN'T seen, to see how good it really is
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
from sklearn.linear_model import Ridge

# Train a second model: Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_predictions = ridge_model.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_rmse = mean_squared_error(y_test, ridge_predictions) ** 0.5
ridge_r2 = r2_score(y_test, ridge_predictions)

print("\nRIDGE REGRESSION PERFORMANCE:")
print(f"MAE: {ridge_mae:.2f}")
print(f"RMSE: {ridge_rmse:.2f}")
print(f"R² Score: {ridge_r2:.2f}")

print("\n--- COMPARISON ---")
print(f"Random Forest R²: {r2:.2f}")
print(f"Ridge Regression R²: {ridge_r2:.2f}")
