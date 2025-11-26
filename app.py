import os
import sys
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- CUSTOM MODULE IMPORT ---
# Expects 'baselinesorption' folder to be in the same directory
try:
    from baselinesorption.baselinesorption import predict_water_yield
    print("✓ AWG Module (baselinesorption) loaded successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'baselinesorption'. Ensure the folder is present. {e}")
    sys.exit(1)

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# --- MODEL LOADING ---
print("--- INITIALIZING AEROAQUA SERVER ---")
print("Loading ML Models...")
try:
    fog_model = joblib.load('models/fog_model.joblib')
    dew_model = joblib.load('models/dew_model.joblib')
    dew_scaler = joblib.load('models/dew_scaler.joblib')
    print("✓ Fog and Dew ML Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")
    sys.exit(1)

# --- OPEN METEO SETUP ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather_data():
    """Fetches raw hourly data required for feature engineering."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 43.7,
        "longitude": -79.4,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_speed_10m", "wind_speed_100m",
            "wind_direction_10m", "wind_direction_100m",
            "surface_pressure", "cloud_cover", "cloud_cover_low",
            "rain", "shortwave_radiation"
        ],
        "forecast_days": 16
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            periods=len(hourly.Variables(0).ValuesAsNumpy()),
            freq="h"
        )}
        
        variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_speed_10m", "wind_speed_100m",
            "wind_direction_10m", "wind_direction_100m",
            "surface_pressure", "cloud_cover", "cloud_cover_low",
            "rain", "shortwave_radiation"
        ]
        
        for i, var_name in enumerate(variables):
            data[var_name] = hourly.Variables(i).ValuesAsNumpy()
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return None

def get_historic_weather_data(start_date, end_date):
    """Fetches raw hourly data from the Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 43.7,
        "longitude": -79.4,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_speed_10m", "wind_speed_100m",
            "wind_direction_10m", "wind_direction_100m",
            "surface_pressure", "cloud_cover", "cloud_cover_low",
            "rain", "shortwave_radiation"
        ]
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            periods=len(hourly.Variables(0).ValuesAsNumpy()),
            freq="h"
        )}
        
        variables = [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_speed_10m", "wind_speed_100m",
            "wind_direction_10m", "wind_direction_100m",
            "surface_pressure", "cloud_cover", "cloud_cover_low",
            "rain", "shortwave_radiation"
        ]
        
        for i, var_name in enumerate(variables):
            data[var_name] = hourly.Variables(i).ValuesAsNumpy()
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching historic weather: {e}")
        return None

# --- FEATURE ENGINEERING ---

def prepare_fog_features(df):
    """Generates the 18 features expected by the Fog Random Forest."""
    for h in [10, 100]:
        rad = np.deg2rad(df[f"wind_direction_{h}m"] % 360)
        df[f"wind_dir_{h}m_sin"] = np.sin(rad)
        df[f"wind_dir_{h}m_cos"] = np.cos(rad)

    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    features = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "wind_speed_100m",
        "wind_dir_10m_sin", "wind_dir_10m_cos",
        "wind_dir_100m_sin", "wind_dir_100m_cos",
        "surface_pressure", "cloud_cover", "cloud_cover_low",
        "rain", "shortwave_radiation",
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    ]
    return df[features]

def predict_dew_yield(day_df):
    """Calculates the 5 features expected by the Dew MLP and predicts."""
    try:
        RH_max = day_df["relative_humidity_2m"].max()
        RH_avg = day_df["relative_humidity_2m"].mean()
        dew_point = day_df["dew_point_2m"].mean()
        e_hPa = 6.112 * np.exp((17.67 * dew_point) / (dew_point + 243.5))
        TI_C = day_df["temperature_2m"].max() - day_df["temperature_2m"].min()
        WV = day_df["wind_speed_10m"].mean()
        
        features = np.array([[RH_max, RH_avg, e_hPa, TI_C, WV]])
        features_scaled = dew_scaler.transform(features)
        pred_mL = dew_model.predict(features_scaled)[0]
        
        return max(0.0, float(pred_mL) / 1000.0)
    except Exception:
        return 0.0

def predict_awg_yield(day_df):
    """
    Uses the custom 'predict_water_yield' function from 'baselinesorption'.
    Inputs:
      - total_solar_kwh: Daily sum of shortwave_radiation (W/m²) / 1000
      - mean_rh: Daily mean relative humidity
    """
    try:
        mean_rh = day_df["relative_humidity_2m"].mean()
        # Sum hourly W/m² -> divide by 1000 to get kWh/m² approx
        total_solar_kwh = day_df["shortwave_radiation"].sum() / 1000.0
        
        # Call the custom library function
        daily_liters = predict_water_yield(
            solar_energy_kwh_m2=total_solar_kwh,
            rh_percent=mean_rh
        )
        
        return max(float(daily_liters), 0.0)
    except Exception as e:
        print(f"AWG Calculation Error: {e}")
        return 0.0

# --- API ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    df = get_weather_data()
    if df is None:
        return jsonify({"error": "Weather data unavailable"}), 500

    results = []
    unique_days = df['date'].dt.date.unique()
    
    for day in unique_days:
        day_mask = df['date'].dt.date == day
        day_df = df[day_mask].copy()
        
        # Fog
        fog_X = prepare_fog_features(day_df)
        hourly_fog_preds = fog_model.predict(fog_X)
        fog_yield = float(np.sum(np.clip(hourly_fog_preds, 0, None)))
        
        # Dew
        dew_yield = float(predict_dew_yield(day_df))
        
        # AWG
        awg_yield = float(predict_awg_yield(day_df))
        
        # Metadata
        rh_mean = float(day_df["relative_humidity_2m"].mean())
        solar_mean = float(day_df["shortwave_radiation"].mean())
        
        # Best Tech
        best_val = max(fog_yield, awg_yield, dew_yield)
        if best_val == fog_yield: best_tech = 'FOG'
        elif best_val == awg_yield: best_tech = 'AWG'
        else: best_tech = 'DEW'

        results.append({
            "date": str(day),
            "fog": round(fog_yield, 2),
            "awg": round(awg_yield, 2),
            "dew": round(dew_yield, 2),
            "humidity": round(rh_mean, 1),
            "solar": round(solar_mean, 0),
            "best_yield": round(best_val, 2),
            "best_tech": best_tech
        })

    return jsonify({"forecast": results})

@app.route('/api/historic', methods=['GET'])
def get_historic():
    start = request.args.get('start')
    end = request.args.get('end')
    
    df = get_historic_weather_data(start, end)
    
    if df is None:
        return jsonify({"error": "Failed to fetch historic data"}), 500
    
    daily_data = []
    total_vol = 0
    fog_events = 0
    peak_yield = 0
    peak_date = ""
    
    unique_days = df['date'].dt.date.unique()
    
    for day in unique_days:
        day_mask = df['date'].dt.date == day
        day_df = df[day_mask].copy()
        
        # Fog
        fog_X = prepare_fog_features(day_df)
        hourly_fog_preds = fog_model.predict(fog_X)
        fog_yield = float(np.sum(np.clip(hourly_fog_preds, 0, None)))
        
        # Dew
        dew_yield = float(predict_dew_yield(day_df))
        
        # AWG
        awg_yield = float(predict_awg_yield(day_df))
        
        total_day = fog_yield + awg_yield + dew_yield
        total_vol += total_day
        
        if total_day > peak_yield:
            peak_yield = total_day
            peak_date = day.strftime('%b %d')
            
        if fog_yield > 5.0:
            fog_events += 1
        
        daily_data.append({
            "date": str(day),
            "fog": round(fog_yield, 2),
            "awg": round(awg_yield, 2),
            "dew": round(dew_yield, 2)
        })

    days_count = len(unique_days)
    avg_daily = total_vol / days_count if days_count > 0 else 0

    return jsonify({
        "summary": {
            "total": round(total_vol, 1),
            "avg": round(avg_daily, 1),
            "peak": round(peak_yield, 1),
            "peak_date": peak_date,
            "fog_events": fog_events,
            "delta": 0
        },
        "daily": daily_data
    })

# --- SIMULATION ENDPOINT ---
@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    data = request.json
    tank = float(data.get('tank', 1000))
    area = float(data.get('area', 10))
    demand = float(data.get('demand', 50))
    
    levels = []
    curr = tank * 0.5
    fails = 0
    harvest = 0
    
    for i in range(365):
        prod = max(0, 3 + np.sin(i*0.017)*2 + np.random.normal(0,1)) * area
        harvest += prod
        curr += prod - demand
        if curr > tank: curr = tank
        if curr < 0: 
            curr = 0
            fails += 1
        levels.append(round(curr, 1))
        
    return jsonify({
        "total_harvest": round(harvest, 0),
        "failures": fails,
        "efficiency": round((1 - fails/365)*100, 1),
        "levels": levels
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port)