# Flood Prediction Using MLP and LSTM

A Data Mining & Machine Learning project for predicting river water levels and classifying flood events in the Manukau area, New Zealand, using Multilayer Perceptron (MLP) and Long Short-Term Memory (LSTM) neural networks.

**Authors:** Antonius Dimitri Adhisatrio · Nyan Tun Aye

---

## Overview

This project uses hourly environmental data spanning 5 years (May 2020 – April 2025) to build and compare two neural network models for river water level regression and flood classification. The best-performing model is then applied to a real-world case study during an ex-tropical cyclone event (18–20 April 2025).

---

## Dataset

| File | Description |
|------|-------------|
| `data/Manukau.csv` | Hourly environmental measurements from the Manukau area |

**Variables in the dataset:**

| Variable | Description |
|----------|-------------|
| `river_water_level` | River water level (meters) — response variable |
| `river_discharge` | River discharge (m³/s) |
| `sports_bowl_rainfall` | Rainfall at Sports Bowl location (mm) |
| `botanical_garden_rainfall` | Rainfall at Botanical Gardens location (mm) |
| `relative_humidity` | Relative humidity (%) |
| `air_temperature` | Air temperature (°C) |
| `wind_speed` | Wind speed (m/s) |
| `wind_direction` | Wind direction (degrees) |

---

## Project Structure

```
flood-prediction/
├── data/
│   └── Manukau.csv
├── flood_prediction.py
└── README.md
```

---

## Methodology

### 1. Data Preprocessing
- Removed duplicate `start_interval` column; set `end_interval` as datetime index
- Verified 1-hour temporal resolution across all 43,800 rows
- Imputed missing values (`relative_humidity`: 2,421 missing; `wind_direction`: 2,421 missing) using time-based interpolation (forward + backward)
- Detected and corrected erroneous outliers in `river_water_level` (e.g. sudden spikes from 0.2m to 20m within one hour) by converting them to NaN and re-imputing
- Retained natural extreme events in `river_discharge` and rainfall variables
- Scaled all features to [0, 1] using **MinMaxScaler**
- Defined flood classification using the **99th percentile** of scaled river water level (438 flood events out of 43,800 total)

### 2. Feature Selection
Selected 4 features based on correlation with `river_water_level`:

| Feature | Correlation |
|---------|-------------|
| `river_discharge` | 0.75 |
| `sports_bowl_rainfall` | 0.37 |
| `botanical_garden_rainfall` | 0.37 |
| `relative_humidity` | 0.34 |

### 3. Sequence Construction & Data Split
- **Lookback window:** 24 hours
- **Forecast horizon:** 3 hours
- **Split:** 70% train / 20% test / 10% validation

### 4. Models

#### MLP (Multilayer Perceptron)
- Tuned learning rate across `[0.001, 0.01, 0.1]` → best: `0.01`
- Tuned neuron split across all combinations of 25 total neurons across 2 hidden layers → best: `8–17`

#### LSTM (Long Short-Term Memory)
Hyperparameter search over 30 runs each:
- **Best epoch:** 28
- **Best batch size:** 64
- **Best neuron combination:** (75, 50) — selected based on lowest validation RMSE

Final LSTM architecture:
```
LSTM(75, return_sequences=True) → Dropout(0.3)
LSTM(50, return_sequences=False) → Dropout(0.2)
Dense(3)
```
Optimizer: Adam (lr=0.01)

---

## Results

### Overall Model Comparison

| Metric | MLP | LSTM |
|--------|-----|------|
| MAE | 0.007 | 0.005 |
| RMSE | 0.013 | 0.012 |
| R² | 0.897 | 0.921 |
| Recall (Flood) | 0.587 | 0.627 |
| F1-Score (Flood) | 0.642 | 0.701 |

**LSTM outperforms MLP across all metrics**, particularly for flood detection recall and F1-score.

### Case Study: Ex-Tropical Cyclone (18–20 April 2025)
The LSTM model closely tracked the actual water level peak (~0.40 scaled) during the cyclone, with only minor deviations in the decline phase. The MLP slightly overpredicted the peak (~0.45 scaled). Both models captured the major trend effectively.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

---

## How to Run

1. Clone the repository and place `Manukau.csv` inside the `data/` folder.
2. Install dependencies (see above).
3. Run the script:

```bash
python flood_prediction.py
```

> **Note:** The LSTM hyperparameter search (30 runs × 3 configurations) is computationally intensive. Running on a GPU or Google Colab is strongly recommended.

---

## Key Findings

- LSTM's ability to capture temporal dependencies makes it better suited for time-series flood prediction than MLP
- Both models degrade in performance as the forecast horizon extends from 1 to 3 hours, but LSTM's advantage becomes more pronounced at longer horizons
- The 99th percentile flood threshold results in a heavily imbalanced dataset (99% non-flood), which limits classification recall — no rebalancing was applied as the response variable is continuous
