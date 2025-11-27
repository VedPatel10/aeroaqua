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
try:
    from baselinesorption.baselinesorption import predict_water_yield
    print("✓ AWG Module (baselinesorption) loaded successfully.")
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 'baselinesorption'. {e}")
    sys.exit(1)

# Initialize Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# --- MODEL LOADING ---
print("--- INITIALIZING AEROAQUA SERVER ---")
try:
    fog_model = joblib.load('models/fog_model_new.joblib')
    dew_model = joblib.load('models/dew_model_new.joblib')
    dew_scaler = joblib.load('models/dew_scaler_new.joblib')
    print("✓ Fog and Dew ML Models loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load models. {e}")
    sys.exit(1)

# --- OPEN METEO SETUP ---
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --- HELPER: DATA FETCHING ---
def get_weather_dataframe(url, params):
    """Generic helper to fetch and format OpenMeteo data"""
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        # Determine frequency (OpenMeteo uses seconds)
        interval = hourly.Interval()
        
        data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            periods=len(hourly.Variables(0).ValuesAsNumpy()),
            freq=f"{interval}s"
        )}
        
        variables = params['hourly']
        for i, var_name in enumerate(variables):
            data[var_name] = hourly.Variables(i).ValuesAsNumpy()
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"API Error: {e}")
        return None

# --- VECTORIZED FEATURE ENGINEERING ---

def prepare_fog_features_vectorized(df):
    """
    Generates features for the ENTIRE dataframe at once.
    Much faster than looping day by day.
    """
    # FIXED: Use Pandas native check instead of NumPy
    # This prevents the "Cannot interpret 'datetime64[ns, UTC]'" error
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Wind Trignometry
    for h in [10, 100]:
        rad = np.deg2rad(df[f"wind_direction_{h}m"] % 360)
        df[f"wind_dir_{h}m_sin"] = np.sin(rad)
        df[f"wind_dir_{h}m_cos"] = np.cos(rad)

    # Time Cycles
    hours = df["date"].dt.hour
    months = df["date"].dt.month
    
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["month_sin"] = np.sin(2 * np.pi * months / 12)
    df["month_cos"] = np.cos(2 * np.pi * months / 12)

    features = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "wind_speed_100m",
        "wind_dir_10m_sin", "wind_dir_10m_cos",
        "wind_dir_100m_sin", "wind_dir_100m_cos",
        "surface_pressure", "cloud_cover", "cloud_cover_low",
        "rain", "shortwave_radiation",
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    ]
    
    # Return just the feature matrix (X)
    return df[features]

def calculate_daily_predictions(hourly_df):
    """
    Performs vectorized predictions for all models.
    """
    # 1. FOG: Predict hourly on the whole dataset first
    # This calls the Random Forest ONCE for 8000+ rows (very fast)
    # instead of calling it 365 times for 24 rows.
    X_fog = prepare_fog_features_vectorized(hourly_df)
    
    # Predict and Clip
    fog_preds = fog_model.predict(X_fog)
    hourly_df['fog_pred'] = np.clip(fog_preds, 0, None)
    
    # 2. AGGREGATE TO DAILY: Create a daily summary dataframe
    # We group by date to get the stats needed for Dew/AWG models
    daily_stats = hourly_df.groupby(hourly_df['date'].dt.date).agg({
        'fog_pred': 'sum',
        'relative_humidity_2m': ['mean', 'max'],
        'dew_point_2m': 'mean',
        'temperature_2m': ['max', 'min'],
        'wind_speed_10m': 'mean',
        'shortwave_radiation': 'sum'
    })
    
    # Flatten MultiIndex columns (e.g., ('temperature_2m', 'max') -> 'temp_max')
    daily_stats.columns = ['_'.join(col).strip() for col in daily_stats.columns.values]
    
    # 3. DEW: Vectorized calculation on Daily Data
    # Calculate physics features for the whole column at once
    dew_point = daily_stats['dew_point_2m_mean']
    # Tetens formula vectorized
    e_hPa = 6.112 * np.exp((17.67 * dew_point) / (dew_point + 243.5))
    
    # Prepare Input Matrix for Dew Model
    # Features: [RH_max, RH_avg, e_hPa, TI_C, WV]
    RH_max = daily_stats['relative_humidity_2m_max']
    RH_avg = daily_stats['relative_humidity_2m_mean']
    TI_C = daily_stats['temperature_2m_max'] - daily_stats['temperature_2m_min']
    WV = daily_stats['wind_speed_10m_mean']
    
    # Create DataFrame for Scaler
    X_dew = pd.DataFrame({
        'RH_max': RH_max,
        'RH_avg': RH_avg,
        'e_hPa': e_hPa,
        'TI_C': TI_C,
        'WV': WV
    })
    
    # Scale and Predict
    X_dew_scaled = dew_scaler.transform(X_dew)
    dew_preds_mL = dew_model.predict(X_dew_scaled)
    daily_stats['dew_pred'] = np.clip(dew_preds_mL / 1000.0, 0, None) # Convert to L
    
    # 4. AWG: Row-by-row application (Custom logic is harder to vectorize fully)
    # But since it's now applied to 365 rows instead of raw loops, it's fast.
    def calc_awg(row):
        try:
            # Solar sum is W/m², divide by 1000 -> kWh/m² approx for the day
            solar_kwh = row['shortwave_radiation_sum'] / 1000.0
            rh_mean = row['relative_humidity_2m_mean']
            return max(0.0, float(predict_water_yield(solar_kwh, rh_mean)))
        except:
            return 0.0

    daily_stats['awg_pred'] = daily_stats.apply(calc_awg, axis=1)
    
    return daily_stats

# --- API ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 43.7, "longitude": -79.4,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
            "surface_pressure", "cloud_cover", "cloud_cover_low", "rain", "shortwave_radiation"
        ],
        "forecast_days": 16
    }
    
    df = get_weather_dataframe(url, params)
    if df is None: return jsonify({"error": "Weather data unavailable"}), 500

    # Run Vectorized Pipeline
    daily_stats = calculate_daily_predictions(df)
    
    # Format Results
    results = []
    # Join with metadata for display
    daily_meta = df.groupby(df['date'].dt.date).agg({
        'relative_humidity_2m': 'mean',
        'shortwave_radiation': 'mean'
    })

    for date, row in daily_stats.iterrows():
        # FIXED: Explicitly cast NumPy float32 to Python float for JSON serialization
        fog_y = float(row['fog_pred_sum'])
        awg_y = float(row['awg_pred'])
        dew_y = float(row['dew_pred'])
        
        # Retrieve display metadata
        rh_mean = float(daily_meta.loc[date, 'relative_humidity_2m'])
        solar_mean = float(daily_meta.loc[date, 'shortwave_radiation'])

        best_val = max(fog_y, awg_y, dew_y)
        if best_val == fog_y: best_tech = 'FOG'
        elif best_val == awg_y: best_tech = 'AWG'
        else: best_tech = 'DEW'

        results.append({
            "date": str(date),
            "fog": round(fog_y, 2),
            "awg": round(awg_y, 2),
            "dew": round(dew_y, 2),
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
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 43.7, "longitude": -79.4,
        "start_date": start, "end_date": end,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
            "surface_pressure", "cloud_cover", "cloud_cover_low", "rain", "shortwave_radiation"
        ]
    }
    
    df = get_weather_dataframe(url, params)
    if df is None: return jsonify({"error": "Failed to fetch historic data"}), 500
    
    # Run Vectorized Pipeline
    daily_stats = calculate_daily_predictions(df)
    
    daily_data = []
    fog_events = 0
    
    for date, row in daily_stats.iterrows():
        # FIXED: Explicitly cast NumPy float32 to Python float here as well
        fog_y = float(row['fog_pred_sum'])
        if fog_y > 5.0: fog_events += 1
        
        daily_data.append({
            "date": str(date),
            "fog": round(fog_y, 2),
            "awg": round(float(row['awg_pred']), 2),
            "dew": round(float(row['dew_pred']), 2)
        })

    return jsonify({
        "summary": { "fog_events": fog_events },
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
    app.run(host='0.0.0.0', port=port)