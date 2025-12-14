# AeroAqua: Optimizing Atmospheric Water Harvesting using Machine Learning

AeroAqua is a machine-learning–driven framework for predicting and optimizing three atmospheric water harvesting (AWH) technologies:

- **Fog Nets** (advective fog capture)
- **Active Atmospheric Water Generators (AWGs)**
- **Dew Condensers** (radiative cooling surfaces)

The project unifies **data processing**, **synthetic data generation**, **feature engineering**, and **ML modeling** to simulate water yields from historical and forecasted meteorology.

Interact with AeroAqua at https://aeroaqua.onrender.com/ to explore multi-day fog, dew, and AWG yield forecasts powered by these models. Note: you may need to wait 1-2 minutes for the webpage to load if it has been inactive for 15+ minutes. 

Here is an example of the 3 models predicting 2016 yields:

<img width="1389" height="590" alt="2016modelyields" src="https://github.com/user-attachments/assets/317eb129-d1ff-4647-8daa-86f0757e8d2e" />

---

## 1. Machine Learning Challenges in Atmospheric Water Harvesting

Atmospheric water harvesting depends on nonlinear interactions between:

- humidity dynamics
- dew point convergence
- wind speed & direction
- cloud cover
- solar radiation cycles
- nighttime radiative cooling

AeroAqua builds **three separate ML models**, each trained on its own reconstructed or synthetic dataset, then combines them into a hybrid system that determines the most effective technology for each day.

---

## 2. Fog Net Model — Random Forest + Synthetic Fog Generation

### 2.1 Raw Fog Data

Fog data is sparse in most climates. We combined:

- MIT-14 fog net yield data ([paper](https://hdl.handle.net/1721.1/119788))
- Pepperwood Preserve fog conditions
- Hourly meteorology from Open-Meteo

However, real fog events were too few to train a robust regressor.

### 2.2 Synthetic Fog Events

To overcome this imbalance, AeroAqua generates synthetic fog samples using:

- **Gaussian Copula Synthesizer** (correlated distributions)
- **CTGAN** (Conditional GAN for rare fog cases)
- **TVAE** (latent-space interpolation)

These models learn multivariate relationships between RH, dew point, wind, cloud cover, and fog yield.

### 2.3 Feature Engineering

Each hourly sample includes:

- temperature_2m
- relative_humidity_2m
- dew_point_2m
- wind_speed_10m / 100m
- wind_dir_10m_sin, wind_dir_10m_cos
- wind_dir_100m_sin, wind_dir_100m_cos
- surface_pressure
- cloud_cover, cloud_cover_low
- rain
- shortwave_radiation
- hour_sin, hour_cos
- month_sin, month_cos

### 2.4 Final Fog Model

A Random Forest predicts **hourly fog yield (L/hr)**:

```
RandomForestRegressor(
    n_estimators=500,
    min_samples_leaf=2,
    random_state=42
)
```

---

## 3. Dew Condenser Model — Histogram Reconstruction + MLP

### 3.1 The Data Problem

Dew studies rarely publish raw datapoints. The Poland dew condenser dataset ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0169809514002075)) only provided:

- histogram buckets
- category frequencies
- aggregate statistics

No `(weather → dew yield)` rows existed.

### 3.2 Bucket → Datapoint Reconstruction

AeroAqua reconstructs synthetic dew datapoints by sampling from:

- RH bucket frequencies
- yield distribution buckets
- wind speed distributions
- temperature amplitude buckets

For each synthetic row we compute:

- **RH_max**
- **RH_avg**
- **temperature amplitude (TI_C)**
- **nighttime wind speed (WV)**
- **vapor pressure using Tetens formula (e_hPa)**

This yields a statistically accurate approximation of the original experiment.

### 3.3 Final Dew Model

A small MLP predicts **nightly dew yield (L/m²)**:

```
MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    alpha=0.001,
    early_stopping=True
)
```

Inputs are scaled via:

---

## 4. AWG Model — Solar Zenith + Physics-Informed Regression

The AWG model computes yield from:

- **solar energy absorbed (kWh/m²)**
- **ambient humidity**

### 4.1 Solar Zenith Pipeline

1. Compute solar zenith angle from latitude, day, and time.
2. Estimate GHI (solar irradiance) from zenith using a learned regression model.
3. Integrate W/m² → **kWh/m² per day**.

### 4.2 Yield Prediction

Using a custom sorption-desorption model:

```python
predict_water_yield(solar_energy_kwh_m2, rh_percent)
```

Outputs daily AWG yield (L/day).

---

## 6. Acknowledgements

This project was done in collaboration with RBC Borealis, specifically the Let's SOLVE It program. 

Team Members: [Ved Patel](https://github.com/VedPatel10), [Gawtham Senthilvelan](https://github.com/Gawthaman), [David Li](https://github.com/0314davidli), and [Karthik Pakki](https://github.com/KarthikPakki)
