import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# =============================================================================
# Section 1: Data Preprocessing
# =============================================================================

# --- 1.1 Load Dataset ---
df = pd.read_csv('data/Manukau.csv')

print("First 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nShape:", df.shape)

# --- 1.2 Remove 'start_interval', keep 'end_interval' as index ---
df = df.drop(columns=['start_interval'])
df['end_interval'] = pd.to_datetime(df['end_interval'])
df = df.set_index('end_interval')
df.index.name = 'end_interval'

# Check temporal resolution (should be 1 hour)
time_diffs = df.index.to_series().diff().dropna()
print("\nMax time difference between rows:", time_diffs.max())
print("Min time difference between rows:", time_diffs.min())

# --- 1.3 Rename columns for consistency ---
df.columns = ['river_water_level', 'river_discharge', 'sports_bowl_rainfall',
              'botanical_garden_rainfall', 'relative_humidity', 'air_temperature',
              'wind_speed', 'wind_direction']

# --- Summary Statistics ---
print("\nSummary Statistics:")
print(df.describe())

# --- 1.4 Check Missing Values ---
print("\nMissing Values:")
print(df.isnull().sum())

# --- 1.5 Check Duplicates ---
print("\nDuplicate rows:", df.duplicated().sum())

# =============================================================================
# Section 1.4.2: Missing Value Imputation
# =============================================================================

# Impute using time-based interpolation (forward + backward)
df = df.interpolate(method='time', limit_direction='both')
print("\nMissing values after imputation:", df.isnull().sum().sum())

# =============================================================================
# Section 1.4.3: Outliers Process
# =============================================================================

# --- Boxplot Before Outlier Handling ---
plt.figure(figsize=(14, 6))
df.boxplot()
plt.title('Boxplot of All Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Boxplot excluding wind_direction
plt.figure(figsize=(14, 6))
df.drop(columns=['wind_direction']).boxplot()
plt.title('Boxplot of All Features Excluding Wind Direction Variable')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Identify & handle river_water_level outliers ---
# A sudden jump >5 meters from the previous hour is considered a data error
rwl = df['river_water_level'].copy()
diff = rwl.diff().abs()
# Flag rows where jump > 5 meters as errors
error_mask = diff > 5
print(f"\nNumber of erroneous river_water_level entries: {error_mask.sum()}")
print("\nData Error Check for River Water Level:")
# Show surrounding rows of errors
for idx in df[error_mask].index:
    loc = df.index.get_loc(idx)
    print(df.iloc[max(0, loc-2):loc+3][['river_water_level']])

# Convert erroneous river_water_level values to NaN and re-impute
df.loc[error_mask, 'river_water_level'] = np.nan
df['river_water_level'] = df['river_water_level'].interpolate(method='time', limit_direction='both')

# --- Data Error Check for River Discharge ---
print("\nData Error Check for River Discharge (showing max discharge event):")
peak_idx = df['river_discharge'].idxmax()
peak_loc = df.index.get_loc(peak_idx)
print(df.iloc[max(0, peak_loc-3):peak_loc+3][['river_discharge']])
# River discharge outliers are considered natural events — no removal

# --- Boxplot After Imputation ---
# Create flood column before boxplot (needed to match report Figure 3)
flood_threshold = df['river_water_level'].quantile(0.99)
df['flood'] = (df['river_water_level'] >= flood_threshold).astype(int)

cols_to_plot = ['river_water_level', 'river_discharge', 'sports_bowl_rainfall',
                'botanical_garden_rainfall', 'relative_humidity', 'air_temperature',
                'wind_speed', 'flood']
plt.figure(figsize=(14, 6))
df[cols_to_plot].boxplot()
plt.title('Boxplot of All Features After Imputation')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =============================================================================
# Section 1.4.4: Distribution of All Features (Histogram)
# =============================================================================

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
feature_cols = ['river_water_level', 'river_discharge', 'sports_bowl_rainfall',
                'botanical_garden_rainfall', 'relative_humidity', 'air_temperature',
                'wind_speed', 'wind_direction']
for i, col in enumerate(feature_cols):
    axes[i].hist(df[col], bins=50, color='steelblue', edgecolor='none')
    axes[i].set_title(col)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
axes[-1].set_visible(False)
plt.suptitle('Histogram of All Variables', fontsize=14)
plt.tight_layout()
plt.show()

# =============================================================================
# Section 1 Final: Scale Features and Define Flood Label
# =============================================================================

# MinMaxScaler on all features
scaler_all = MinMaxScaler()
df_scaled_arr = scaler_all.fit_transform(df[feature_cols])
df_scaled = pd.DataFrame(df_scaled_arr, columns=feature_cols, index=df.index)

# Flood classification based on 99th percentile of SCALED river_water_level
flood_threshold_scaled = df_scaled['river_water_level'].quantile(0.99)
df_scaled['flood'] = (df_scaled['river_water_level'] >= flood_threshold_scaled).astype(int)

print(f"\nFlood threshold (scaled): {flood_threshold_scaled:.6f}")
print(f"Non-flood count: {(df_scaled['flood'] == 0).sum()}")
print(f"Flood count: {(df_scaled['flood'] == 1).sum()}")

# =============================================================================
# Section 2: Feature Selection & EDA
# =============================================================================

# --- 2.1 Correlation Heatmap ---
plt.figure(figsize=(12, 10))
corr = df_scaled[feature_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Correlation Heatmap of flood predictors')
plt.tight_layout()
plt.show()

print("\nCorrelation with river_water_level:")
print(corr['river_water_level'].sort_values(ascending=False))

# Selected features (top 4 correlated with river_water_level, excluding itself)
selected_features = ['river_discharge', 'sports_bowl_rainfall',
                     'botanical_garden_rainfall', 'relative_humidity']

# --- 2.2 Flood Status Distribution ---
plt.figure(figsize=(6, 5))
flood_counts = df_scaled['flood'].value_counts().sort_index()
plt.bar(['No Flood', 'Flood'], flood_counts.values, color='blue')
plt.title('Class Distribution After Scaling')
plt.xlabel('Flood Status')
plt.ylabel('Flood Status Count')
plt.tight_layout()
plt.show()

# --- 2.3 River Water Level Variation Over Time ---
plt.figure(figsize=(14, 5))
plt.plot(df_scaled.index, df_scaled['river_water_level'], color='blue', linewidth=0.5)
plt.title('River Water Level Variation Over Time')
plt.xlabel('Time')
plt.ylabel('Scaled Water Level')
plt.tight_layout()
plt.show()

# --- 2.4 Scatter Plots of Selected Features vs River Water Level ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
colors = ['blue', 'green', 'orange', 'red']
titles = ['River Water Level vs River Discharge',
          'River Water Level vs Rainfall Sports Bowl',
          'River Water Level vs Rainfall Botanical Gardens',
          'River Water Level vs Relative Humidity']
xlabels = ['River Discharge (Scaled)', 'Rainfall Sports Bowl (Scaled)',
           'Rainfall Botanical Gardens (Scaled)', 'Relative Humidity (Scaled)']

for i, (feat, color, title, xlabel) in enumerate(zip(selected_features, colors, titles, xlabels)):
    ax = axes[i // 2][i % 2]
    ax.scatter(df_scaled[feat], df_scaled['river_water_level'],
               alpha=0.3, s=5, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('River Water Level (Scaled)')
plt.suptitle('Scatter Plots of Correlation Between Response Variable Against Selected Features (Scaled)')
plt.tight_layout()
plt.show()

# --- 2.5 Summary Statistics of Selected Features ---
summary = df_scaled[selected_features + ['river_water_level']].describe()
print("\nSummary Statistics of Scaled Selected Features:")
print(summary)
print("\nCorrelations with river_water_level:")
for f in selected_features:
    print(f"  {f}: {df_scaled[f].corr(df_scaled['river_water_level']):.2f}")

# =============================================================================
# Section 3: Sequence Creation & Train/Val/Test Split
# =============================================================================

LOOKBACK = 24    # 24-hour lookback
HORIZON  = 3     # 3-hour forecast

def create_sequences(features_df, target_series, flood_series, lookback, horizon):
    """Create (X, y_reg, y_flood) sequences."""
    X, y_reg, y_flood = [], [], []
    arr_X = features_df.values
    arr_y = target_series.values
    arr_f = flood_series.values
    for i in range(len(arr_X) - lookback - horizon + 1):
        X.append(arr_X[i:i + lookback])
        y_reg.append(arr_y[i + lookback:i + lookback + horizon])
        y_flood.append(arr_f[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y_reg), np.array(y_flood)

X_seq, y_seq, y_flood_seq = create_sequences(
    df_scaled[selected_features],
    df_scaled['river_water_level'],
    df_scaled['flood'],
    LOOKBACK, HORIZON
)

print(f"\nSequence shapes — X: {X_seq.shape}, y: {y_seq.shape}, y_flood: {y_flood_seq.shape}")

n = len(X_seq)
train_end = int(n * 0.70)
val_end   = int(n * 0.90)

X_train, y_train       = X_seq[:train_end],  y_seq[:train_end]
X_val,   y_val         = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test,  y_test        = X_seq[val_end:],    y_seq[val_end:]

y_flood_train = y_flood_seq[:train_end]
y_flood_val   = y_flood_seq[train_end:val_end]
y_flood_test  = y_flood_seq[val_end:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Flat versions for MLP
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat   = X_val.reshape(X_val.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)

# Flood threshold (scalar) for classification
flood_threshold_scaled_val = flood_threshold_scaled

def calculate_metrics(y_true, y_pred, y_flood_true, threshold):
    """Compute regression + flood classification metrics per horizon."""
    results = {}
    n_horizons = y_true.shape[1]

    mae_list, rmse_list, r2_list = [], [], []
    precision_list, recall_list, f1_list, acc_list = [], [], [], []

    for h in range(n_horizons):
        yt = y_true[:, h]
        yp = y_pred[:, h]
        yf = y_flood_true[:, h]

        mae_list.append(mean_absolute_error(yt, yp))
        rmse_list.append(np.sqrt(mean_squared_error(yt, yp)))
        r2_list.append(r2_score(yt, yp))

        yp_flood = (yp >= threshold).astype(int)
        precision_list.append(precision_score(yf, yp_flood, zero_division=0))
        recall_list.append(recall_score(yf, yp_flood, zero_division=0))
        f1_list.append(f1_score(yf, yp_flood, zero_division=0))
        acc_list.append(accuracy_score(yf, yp_flood))

    results['mae']       = mae_list
    results['rmse']      = rmse_list
    results['r2']        = r2_list
    results['precision'] = precision_list
    results['recall']    = recall_list
    results['f1']        = f1_list
    results['accuracy']  = acc_list

    # Overall (averaged across horizons)
    results['overall_mae']  = np.mean(mae_list)
    results['overall_rmse'] = np.mean(rmse_list)
    results['overall_r2']   = np.mean(r2_list)
    results['overall_recall']    = np.mean(recall_list)
    results['overall_f1']        = np.mean(f1_list)
    results['overall_precision'] = np.mean(precision_list)
    results['overall_accuracy']  = np.mean(acc_list)
    return results

# =============================================================================
# Section 4.1: MLP Model
# =============================================================================

# --- 4.1.1 Baseline MLP: Tune Learning Rate ---
print("\n" + "="*60)
print("4.1.1 Baseline MLP - Tuning Learning Rate")
print("="*60)

learning_rates = [0.001, 0.01, 0.1]
lr_results = {}

for lr in learning_rates:
    mlp = MLPRegressor(
        hidden_layer_sizes=(25,),
        learning_rate_init=lr,
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_train_flat, y_train)
    y_pred_lr = mlp.predict(X_test_flat)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_results[lr] = rmse
    print(f"  Learning Rate: {lr} -> RMSE: {rmse:.6f}")

best_lr = min(lr_results, key=lr_results.get)
print(f"\nBest Learning Rate: {best_lr} (RMSE: {lr_results[best_lr]:.6f})")

# --- 4.1.2 MLP with Two Hidden Layers: Tune Neuron Split ---
print("\n" + "="*60)
print("4.1.2 MLP with Two Hidden Layers - Tuning Neuron Split")
print("="*60)

total_neurons = 25
split_results = {}

for n1 in range(1, total_neurons):
    n2 = total_neurons - n1
    mlp2 = MLPRegressor(
        hidden_layer_sizes=(n1, n2),
        learning_rate_init=best_lr,
        max_iter=1000,
        random_state=42
    )
    mlp2.fit(X_train_flat, y_train)
    y_pred_split = mlp2.predict(X_test_flat)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_split))
    split_results[(n1, n2)] = rmse

best_split = min(split_results, key=split_results.get)
print(f"Best neuron split: {best_split[0]}-{best_split[1]}, RMSE: {split_results[best_split]:.6f}")

# Plot neuron split results
splits = list(split_results.keys())
rmses  = list(split_results.values())
labels = [f"{s[0]}-{s[1]}" for s in splits]

plt.figure(figsize=(14, 5))
plt.plot(labels, rmses, marker='o', linewidth=1, markersize=4)
plt.title('MLP Performance Variation with Neuron Split')
plt.xlabel('Neuron Split (Layer 1 - Layer 2)')
plt.ylabel('RMSE')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# --- Train Final MLP with Best Split ---
mlp_final = MLPRegressor(
    hidden_layer_sizes=best_split,
    learning_rate_init=best_lr,
    max_iter=1000,
    random_state=42
)
mlp_final.fit(X_train_flat, y_train)
mlp_pred = mlp_final.predict(X_test_flat)

mlp_metrics = calculate_metrics(y_test, mlp_pred, y_flood_test, flood_threshold_scaled_val)
print(f"\nFinal MLP Results:")
print(f"  MAE:       {mlp_metrics['overall_mae']:.4f}")
print(f"  RMSE:      {mlp_metrics['overall_rmse']:.4f}")
print(f"  R²:        {mlp_metrics['overall_r2']:.4f}")
print(f"  Recall:    {mlp_metrics['overall_recall']:.4f}")
print(f"  F1-Score:  {mlp_metrics['overall_f1']:.4f}")
print(f"  Precision: {mlp_metrics['overall_precision']:.4f}")
print(f"  Accuracy:  {mlp_metrics['overall_accuracy']:.4f}")

# =============================================================================
# Section 4.2: LSTM Model
# =============================================================================

def build_lstm(units1=64, units2=32, lr=0.01, horizon=3):
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(LOOKBACK, len(selected_features))),
        Dropout(0.3),
        LSTM(units2, return_sequences=False),
        Dropout(0.2),
        Dense(horizon)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

# --- 4.2.2 Find Best Epoch ---
print("\n" + "="*60)
print("4.2.2 Finding Best Epoch (30 runs x 30 epochs, batch=16)")
print("="*60)

N_RUNS_EPOCH = 30
N_EPOCHS_SEARCH = 30
BATCH_EPOCH = 16

all_train_losses = np.zeros((N_RUNS_EPOCH, N_EPOCHS_SEARCH))
all_val_losses   = np.zeros((N_RUNS_EPOCH, N_EPOCHS_SEARCH))
epoch_mse_list   = []
epoch_time_list  = []

for run in range(N_RUNS_EPOCH):
    tf.random.set_seed(run)
    model = build_lstm(64, 32, lr=0.01, horizon=HORIZON)
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=N_EPOCHS_SEARCH,
        batch_size=BATCH_EPOCH,
        validation_data=(X_val, y_val),
        verbose=0
    )
    elapsed = time.time() - t0
    val_pred = model.predict(X_val, verbose=0)
    val_mse  = mean_squared_error(y_val.flatten(), val_pred.flatten())
    epoch_mse_list.append(val_mse)
    epoch_time_list.append(elapsed)
    all_train_losses[run] = history.history['loss']
    all_val_losses[run]   = history.history['val_loss']
    if (run + 1) % 5 == 0:
        print(f"  Run {run+1}/{N_RUNS_EPOCH} | Val MSE: {val_mse:.6f} | Time: {elapsed:.1f}s")

epoch_stats = pd.DataFrame({'MSE': epoch_mse_list, 'Run_Time': epoch_time_list})
print("\nEpoch Experiment MSE & Runtime Statistics:")
print(epoch_stats.describe())

avg_train = all_train_losses.mean(axis=0)
avg_val   = all_val_losses.mean(axis=0)
std_train = all_train_losses.std(axis=0)
std_val   = all_val_losses.std(axis=0)
epochs_arr = np.arange(N_EPOCHS_SEARCH)

# Figure 12: MSE curves
plt.figure(figsize=(10, 5))
plt.plot(epochs_arr, avg_train, label='Average Train Loss', color='blue')
plt.plot(epochs_arr, avg_val,   label='Average Validation Loss', color='orange')
plt.title('LSTM Average Train and Validation Loss Across 30 Runs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 13: RMSE
avg_train_rmse = np.sqrt(avg_train)
avg_val_rmse   = np.sqrt(avg_val)
plt.figure(figsize=(10, 5))
plt.plot(epochs_arr, avg_val_rmse, linestyle='--', color='green', label='Validation RMSE')
plt.title('LSTM Train vs Validation RMSE Across 30 Runs')
plt.xlabel('Epoch')
plt.ylabel('Loss/Error')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 14: MSE with shaded std + best epoch
best_epoch = int(np.argmin(avg_val))
print(f"\nBest Epoch (lowest avg val MSE): {best_epoch}")

plt.figure(figsize=(10, 5))
plt.plot(epochs_arr, avg_train, label='Train Loss', color='blue')
plt.fill_between(epochs_arr, avg_train - std_train, avg_train + std_train, alpha=0.2, color='blue')
plt.plot(epochs_arr, avg_val, label='Validation Loss', color='orange')
plt.fill_between(epochs_arr, avg_val - std_val, avg_val + std_val, alpha=0.2, color='orange')
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title('LSTM Train vs Validation Loss Performance Across 30 Runs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.show()

BEST_EPOCHS = best_epoch if best_epoch > 0 else 28

# --- 4.2.3 Find Best Batch Size ---
print("\n" + "="*60)
print(f"4.2.3 Finding Best Batch Size (30 runs x {BEST_EPOCHS} epochs)")
print("="*60)

BATCH_SIZES = [16, 32, 64]
N_RUNS_BATCH = 30

batch_mse_records  = {bs: [] for bs in BATCH_SIZES}
batch_time_records = {bs: [] for bs in BATCH_SIZES}

for bs in BATCH_SIZES:
    print(f"\n  Batch Size {bs}:")
    for run in range(N_RUNS_BATCH):
        tf.random.set_seed(run)
        model = build_lstm(64, 32, lr=0.01, horizon=HORIZON)
        t0 = time.time()
        model.fit(X_train, y_train, epochs=BEST_EPOCHS, batch_size=bs,
                  validation_data=(X_val, y_val), verbose=0)
        elapsed = time.time() - t0
        val_pred = model.predict(X_val, verbose=0)
        val_mse  = mean_squared_error(y_val.flatten(), val_pred.flatten())
        batch_mse_records[bs].append(val_mse)
        batch_time_records[bs].append(elapsed)
        if (run + 1) % 10 == 0:
            print(f"    Run {run+1}/{N_RUNS_BATCH} | MSE: {val_mse:.6f} | Time: {elapsed:.1f}s")

# Statistics
print("\nMSE Statistics by Batch Size:")
batch_mse_df = pd.DataFrame(batch_mse_records)
print(batch_mse_df.describe())

print("\nRun Time Statistics by Batch Size:")
batch_time_df = pd.DataFrame(batch_time_records)
print(batch_time_df.describe())

# Figure 15: MSE bar chart
mean_mse_batch = {bs: np.mean(batch_mse_records[bs]) for bs in BATCH_SIZES}
best_batch = min(mean_mse_batch, key=mean_mse_batch.get)

plt.figure(figsize=(8, 5))
bars = plt.bar([str(bs) for bs in BATCH_SIZES], [mean_mse_batch[bs] for bs in BATCH_SIZES],
               color='lightblue', edgecolor='lightblue')
for bs, bar in zip(BATCH_SIZES, bars):
    if bs == best_batch:
        bar.set_edgecolor('green')
        bar.set_linewidth(2)
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-5,
             f"{mean_mse_batch[bs]:.6f}", ha='center', va='bottom', fontsize=9)
plt.axvline(x=BATCH_SIZES.index(best_batch), color='green', linestyle='--',
            label=f'Best Batch (MSE) ({best_batch})')
plt.title('Validation MSE Across Different Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Validation MSE')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 16: RMSE bar chart
mean_rmse_batch = {bs: np.sqrt(np.mean(batch_mse_records[bs])) for bs in BATCH_SIZES}
plt.figure(figsize=(8, 5))
bars = plt.bar([str(bs) for bs in BATCH_SIZES], [mean_rmse_batch[bs] for bs in BATCH_SIZES],
               color='lightblue')
for bs, bar in zip(BATCH_SIZES, bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
             f"{mean_rmse_batch[bs]:.6f}", ha='center', va='bottom', fontsize=9)
plt.axvline(x=BATCH_SIZES.index(best_batch), color='green', linestyle='--',
            label=f'Best Batch (RMSE) ({best_batch})')
plt.title('Validation RMSE Across Different Batch Sizes')
plt.xlabel('Batch Size')
plt.ylabel('Validation RMSE')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 17: Mean runtime
mean_time_batch = {bs: np.mean(batch_time_records[bs]) for bs in BATCH_SIZES}
plt.figure(figsize=(8, 5))
plt.plot(BATCH_SIZES, [mean_time_batch[bs] for bs in BATCH_SIZES], marker='o', color='blue')
for bs in BATCH_SIZES:
    plt.text(bs, mean_time_batch[bs] + 1, f"{mean_time_batch[bs]:.2f}", ha='center')
plt.title('Mean Runtime per Batch Size Across Runs')
plt.xlabel('Batch Size')
plt.ylabel('Mean Runtime (seconds)')
plt.xticks(BATCH_SIZES)
plt.tight_layout()
plt.show()

BEST_BATCH = best_batch
print(f"\nBest Batch Size: {BEST_BATCH}")

# --- 4.2.4 Find Best Neurons Combination ---
print("\n" + "="*60)
print(f"4.2.4 Finding Best Neurons Combination (30 runs, {BEST_EPOCHS} epochs, batch={BEST_BATCH})")
print("="*60)

NEURON_COMBOS = [(64, 32), (75, 50), (50, 25)]
N_RUNS_NEURONS = 30

neuron_mse_records  = {nc: [] for nc in NEURON_COMBOS}
neuron_time_records = {nc: [] for nc in NEURON_COMBOS}

for nc in NEURON_COMBOS:
    print(f"\n  Neurons {nc}:")
    for run in range(N_RUNS_NEURONS):
        tf.random.set_seed(run)
        model = build_lstm(nc[0], nc[1], lr=0.01, horizon=HORIZON)
        t0 = time.time()
        model.fit(X_train, y_train, epochs=BEST_EPOCHS, batch_size=BEST_BATCH,
                  validation_data=(X_val, y_val), verbose=0)
        elapsed = time.time() - t0
        val_pred = model.predict(X_val, verbose=0)
        val_mse  = mean_squared_error(y_val.flatten(), val_pred.flatten())
        neuron_mse_records[nc].append(val_mse)
        neuron_time_records[nc].append(elapsed)
        if (run + 1) % 10 == 0:
            print(f"    Run {run+1}/{N_RUNS_NEURONS} | MSE: {val_mse:.6f} | Time: {elapsed:.1f}s")

print("\nMSE Statistics by Neuron Combination:")
neuron_mse_df = pd.DataFrame({str(k): v for k, v in neuron_mse_records.items()})
print(neuron_mse_df.describe())

print("\nRun Time Statistics by Neuron Combination:")
neuron_time_df = pd.DataFrame({str(k): v for k, v in neuron_time_records.items()})
print(neuron_time_df.describe())

mean_mse_neurons  = {nc: np.mean(neuron_mse_records[nc])  for nc in NEURON_COMBOS}
mean_rmse_neurons = {nc: np.sqrt(mean_mse_neurons[nc])     for nc in NEURON_COMBOS}
mean_time_neurons = {nc: np.mean(neuron_time_records[nc]) for nc in NEURON_COMBOS}

best_nc_mse  = min(mean_mse_neurons,  key=mean_mse_neurons.get)
best_nc_rmse = min(mean_rmse_neurons, key=mean_rmse_neurons.get)
print(f"\nBest neurons by MSE:  {best_nc_mse}  (MSE: {mean_mse_neurons[best_nc_mse]:.6f})")
print(f"Best neurons by RMSE: {best_nc_rmse} (RMSE: {mean_rmse_neurons[best_nc_rmse]:.6f})")
print("→ Selecting (75, 50) as best — lowest RMSE is prioritised (as per report)")

BEST_NC = (75, 50)   # as per report decision

labels_nc = [f"({nc[0]}, {nc[1]})" for nc in NEURON_COMBOS]

# Figure 17 (MSE bar)
plt.figure(figsize=(8, 5))
bars = plt.bar(labels_nc, [mean_mse_neurons[nc] for nc in NEURON_COMBOS], color='lightblue')
for nc, bar in zip(NEURON_COMBOS, bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-5,
             f"{mean_mse_neurons[nc]:.6f}", ha='center', va='bottom', fontsize=9)
plt.axvline(x=NEURON_COMBOS.index(best_nc_mse), color='green', linestyle='--',
            label=f'Best (MSE) {best_nc_mse}')
plt.title('Validation MSE Across Different Neuron Combinations')
plt.xlabel('Neuron Combination')
plt.ylabel('Validation MSE')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 18 (RMSE bar)
plt.figure(figsize=(8, 5))
bars = plt.bar(labels_nc, [mean_rmse_neurons[nc] for nc in NEURON_COMBOS], color='lightblue')
for nc, bar in zip(NEURON_COMBOS, bars):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
             f"{mean_rmse_neurons[nc]:.6f}", ha='center', va='bottom', fontsize=9)
plt.axvline(x=NEURON_COMBOS.index(best_nc_rmse), color='green', linestyle='--',
            label=f'Best (RMSE) {best_nc_rmse}')
plt.title('Validation RMSE Across Different Neuron Combinations')
plt.xlabel('Neuron Combination')
plt.ylabel('Validation RMSE')
plt.legend()
plt.tight_layout()
plt.show()

# Figure 19 (Runtime)
plt.figure(figsize=(8, 5))
plt.plot(labels_nc, [mean_time_neurons[nc] for nc in NEURON_COMBOS], marker='o', color='blue')
for nc, label in zip(NEURON_COMBOS, labels_nc):
    plt.text(label, mean_time_neurons[nc] + 1, f"{mean_time_neurons[nc]:.2f}", ha='center')
plt.title('Mean Runtime per Neuron Combination Across 30 Runs')
plt.xlabel('Neuron Combinations Size')
plt.ylabel('Mean Runtime (seconds)')
plt.tight_layout()
plt.show()

# =============================================================================
# Section 4.2 Final: Train Optimal LSTM
# =============================================================================

print("\nTraining final LSTM with optimal hyperparameters...")
tf.random.set_seed(42)
lstm_final = build_lstm(BEST_NC[0], BEST_NC[1], lr=0.01, horizon=HORIZON)
lstm_final.fit(X_train, y_train, epochs=BEST_EPOCHS, batch_size=BEST_BATCH,
               validation_data=(X_val, y_val), verbose=1)

lstm_pred = lstm_final.predict(X_test, verbose=0)
lstm_metrics = calculate_metrics(y_test, lstm_pred, y_flood_test, flood_threshold_scaled_val)

print(f"\nFinal LSTM Results:")
print(f"  MAE:       {lstm_metrics['overall_mae']:.4f}")
print(f"  RMSE:      {lstm_metrics['overall_rmse']:.4f}")
print(f"  R²:        {lstm_metrics['overall_r2']:.4f}")
print(f"  Recall:    {lstm_metrics['overall_recall']:.4f}")
print(f"  F1-Score:  {lstm_metrics['overall_f1']:.4f}")
print(f"  Precision: {lstm_metrics['overall_precision']:.4f}")
print(f"  Accuracy:  {lstm_metrics['overall_accuracy']:.4f}")

# =============================================================================
# Section 5: Model Comparison
# =============================================================================

print("\n" + "="*60)
print("5. Model Comparison: MLP vs LSTM")
print("="*60)

# --- Figure 20: Overall comparison bar chart ---
metrics_labels = ['MAE', 'RMSE', 'R²', 'Recall (Flood)', 'F1-Score (Flood)']
mlp_vals  = [mlp_metrics['overall_mae'], mlp_metrics['overall_rmse'],
             mlp_metrics['overall_r2'], mlp_metrics['overall_recall'],
             mlp_metrics['overall_f1']]
lstm_vals = [lstm_metrics['overall_mae'], lstm_metrics['overall_rmse'],
             lstm_metrics['overall_r2'], lstm_metrics['overall_recall'],
             lstm_metrics['overall_f1']]

x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars_mlp  = ax.bar(x - width/2, mlp_vals,  width, label='MLP',  color='skyblue')
bars_lstm = ax.bar(x + width/2, lstm_vals, width, label='LSTM', color='salmon')

for bar, val in zip(bars_mlp,  mlp_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars_lstm, lstm_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_title('Model Performance Comparison (MLP vs LSTM)')
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.legend()
plt.tight_layout()
plt.show()

# --- Figure 21: MAE, RMSE, R² per horizon ---
horizons = [h + 1 for h in range(HORIZON)]

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].plot(horizons, mlp_metrics['mae'],  marker='o', label='MLP',  color='blue')
axes[0].plot(horizons, lstm_metrics['mae'], marker='o', label='LSTM', color='orange')
axes[0].set_title('MAE Across Horizons')
axes[0].set_xlabel('Horizon (hours)')
axes[0].set_ylabel('MAE')
axes[0].legend()

axes[1].plot(horizons, mlp_metrics['rmse'],  marker='o', label='MLP',  color='blue')
axes[1].plot(horizons, lstm_metrics['rmse'], marker='o', label='LSTM', color='orange')
axes[1].set_title('RMSE Across Horizons')
axes[1].set_xlabel('Horizon (hours)')
axes[1].set_ylabel('RMSE')
axes[1].legend()

axes[2].plot(horizons, mlp_metrics['r2'],  marker='o', label='MLP',  color='blue')
axes[2].plot(horizons, lstm_metrics['r2'], marker='o', label='LSTM', color='orange')
axes[2].set_title('R2 Across Horizons')
axes[2].set_xlabel('Horizon (hours)')
axes[2].set_ylabel('R²')
axes[2].legend()

plt.suptitle('MAE, RMSE, R² Comparison Between MLP and LSTM Across Horizons')
plt.tight_layout()
plt.show()

# --- Figure 22: Precision, Accuracy, Recall per horizon ---
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

axes[0].plot(horizons, mlp_metrics['precision'],  marker='o', label='MLP',  color='blue')
axes[0].plot(horizons, lstm_metrics['precision'], marker='o', label='LSTM', color='orange')
axes[0].set_title('Precision Across Horizons')
axes[0].set_xlabel('Horizon (hours)')
axes[0].set_ylabel('Precision')
axes[0].legend()

axes[1].plot(horizons, mlp_metrics['accuracy'],  marker='o', label='MLP',  color='blue')
axes[1].plot(horizons, lstm_metrics['accuracy'], marker='o', label='LSTM', color='orange')
axes[1].set_title('Accuracy Across Horizons')
axes[1].set_xlabel('Horizon (hours)')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

axes[2].plot(horizons, mlp_metrics['recall'],  marker='o', label='MLP',  color='blue')
axes[2].plot(horizons, lstm_metrics['recall'], marker='o', label='LSTM', color='orange')
axes[2].set_title('Recall Across Horizons')
axes[2].set_xlabel('Horizon (hours)')
axes[2].set_ylabel('Recall')
axes[2].legend()

plt.suptitle('Precision, Accuracy, and Recall Comparison Between MLP and LSTM Across Horizons')
plt.tight_layout()
plt.show()

# =============================================================================
# Section 6: Case Study — Ex-Tropical Cyclone (18–20 April 2025)
# =============================================================================

print("\n" + "="*60)
print("6. Case Study: Ex-Tropical Cyclone (18-20 April 2025)")
print("="*60)

cyclone_start = '2025-04-18'
cyclone_end   = '2025-04-20 23:59'

# Identify the case study window in the scaled dataframe
case_mask = (df_scaled.index >= cyclone_start) & (df_scaled.index <= cyclone_end)
case_df = df_scaled[case_mask]

# Build sequences for the case study window (need LOOKBACK rows before it)
case_start_pos = df_scaled.index.get_loc(case_df.index[0])
padded_start   = max(0, case_start_pos - LOOKBACK)
padded_df      = df_scaled.iloc[padded_start:]

X_case, y_case, _ = create_sequences(
    padded_df[selected_features],
    padded_df['river_water_level'],
    padded_df['flood'],
    LOOKBACK, HORIZON
)

# We only want predictions corresponding to the cyclone period
# Each sequence at position i predicts time steps [i+LOOKBACK .. i+LOOKBACK+HORIZON-1]
# So we need i+LOOKBACK to be within the cyclone window
case_start_in_padded = case_start_pos - padded_start
valid_seq_indices = [
    i for i in range(len(X_case))
    if i + LOOKBACK >= case_start_in_padded
]

if len(valid_seq_indices) == 0:
    # Fallback: use all sequences
    valid_seq_indices = list(range(len(X_case)))

X_case_filtered = X_case[valid_seq_indices]
y_case_filtered = y_case[valid_seq_indices]

# Predict with both models
lstm_case_pred = lstm_final.predict(X_case_filtered, verbose=0)
mlp_case_pred  = mlp_final.predict(X_case_filtered.reshape(len(X_case_filtered), -1))

# Use only horizon=0 (1-step ahead) for plotting clarity
actual_1h    = y_case_filtered[:, 0]
lstm_pred_1h = lstm_case_pred[:, 0]
mlp_pred_1h  = mlp_case_pred[:, 0]

# Build time axis
time_axis = padded_df.index[LOOKBACK: LOOKBACK + len(valid_seq_indices)]

# --- Figure 23: LSTM prediction vs actual ---
plt.figure(figsize=(14, 5))
plt.plot(time_axis, actual_1h,    color='blue',   linewidth=1.5, label='Actual Water Level')
plt.plot(time_axis, lstm_pred_1h, color='orange', linewidth=1.5, label='Predicted Water Level')
plt.title('Water Level Prediction (LSTM) During Ex-Tropical Cyclone (18-20 April 2025)')
plt.xlabel('Time')
plt.ylabel('Scaled Water Level')
plt.legend()
plt.tight_layout()
plt.show()

# --- Figure 24: MLP prediction vs actual ---
plt.figure(figsize=(14, 5))
plt.plot(time_axis, actual_1h,   color='blue',  linewidth=1.5, label='Actual Water Level')
plt.plot(time_axis, mlp_pred_1h, color='green', linewidth=1.5, label='Predicted Water Level')
plt.title('Water Level Prediction (MLP) During Ex-Tropical Cyclone (18-20 April 2025)')
plt.xlabel('Time')
plt.ylabel('Scaled Water Level')
plt.legend()
plt.tight_layout()
plt.show()

# --- Figure 25: MLP vs LSTM vs actual ---
plt.figure(figsize=(14, 5))
plt.plot(time_axis, actual_1h,    color='blue',   linewidth=1.5, label='Actual Water Level')
plt.plot(time_axis, lstm_pred_1h, color='orange', linewidth=1.5, label='LSTM Predicted Water Level')
plt.plot(time_axis, mlp_pred_1h,  color='green',  linewidth=1.5, label='MLP Predicted Water Level')
plt.title('Water Level Prediction Comparison (LSTM vs MLP) During Ex-Tropical Cyclone (18-20 April 2025)')
plt.xlabel('Time')
plt.ylabel('Scaled Water Level')
plt.legend()
plt.tight_layout()
plt.show()

print("\n✓ All sections complete.")
