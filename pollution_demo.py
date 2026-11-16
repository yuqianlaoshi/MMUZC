"""Usage: python pollution_demo.py"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'geo_embedding'))

SEG_CSV = os.path.join(DATA_DIR, '04271_seg_results.csv')
SENSOR_CSV = os.path.join(DATA_DIR, 'sensor04271.csv')
ZONE_CSV = os.path.join(DATA_DIR, 'functional_zone_labels.csv')

SEG_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "sky", "car", "truck", "bus",
]

ZONE_MAP = {
    "A": "公共服务", "B": "商业", "G": "绿地与广场",
    "M": "施工", "R": "居住", "S": "交通", "U": "工业",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
KEEP_FRAC = 0.6

print("=" * 60)
print("1. Loading data")
print("=" * 60)

seg = pd.read_csv(SEG_CSV)
print(f"  seg: {seg.shape}")

sensor_raw = pd.read_csv(SENSOR_CSV)
sensor_raw.columns = [
    "Time", "Lat", "Lon", "sol_status",
    "AIR_TEMP", "AIR_HUMIDITY", "AIR_PRESSURE",
    "AIR_WIND_SPEED", "AIR_WIND_SCALE", "AIR_WIND_COMPASS", "AIR_WIND_DIRECTION",
    "AIR_PM1_0", "AIR_PM2_5", "AIR_PM10",
    "GAS_CO", "GAS_SO2", "GAS_NO", "GAS_NO2", "GAS_NH3", "GAS_O3",
]
sensor = sensor_raw
print(f"  sensor: {sensor.shape}")

WEATHER_COLS = [
    "AIR_TEMP", "AIR_HUMIDITY", "AIR_PRESSURE",
    "AIR_WIND_SPEED", "AIR_WIND_SCALE", "AIR_WIND_COMPASS", "AIR_WIND_DIRECTION",
]
POLLUTANTS = ["AIR_PM1_0", "AIR_PM2_5", "AIR_PM10", "GAS_CO", "GAS_SO2", "GAS_NO2"]

n_keep = 8896
seg = seg.iloc[:n_keep]
sensor = sensor.iloc[:n_keep]
print(f"  trimmed to {n_keep} rows")

zone_df = pd.read_csv(ZONE_CSV)
print(f"  zone labels: {zone_df.shape}")

feat_seg = seg[SEG_CLASSES].values.astype(np.float32)
feat_weather = np.nan_to_num(
    sensor[WEATHER_COLS].values.astype(np.float32), nan=0.0)

X = np.concatenate([feat_seg, feat_weather], axis=1)
y = sensor[POLLUTANTS].values.astype(np.float32)
zones = zone_df["zone"].values[:n_keep]

feature_names = SEG_CLASSES + WEATHER_COLS
target_names = POLLUTANTS

print(f"\n  X: {X.shape}  y: {y.shape}  zones: {len(zones)}")

print("\n" + "=" * 60)
print("2. Zone distribution & balanced sampling")
print("=" * 60)

zone_counts = pd.Series(zones).value_counts().sort_index()
for z in sorted(zone_counts.index):
    print(f"    {z} ({ZONE_MAP.get(z, '?')}): {zone_counts[z]}")

np.random.seed(RANDOM_STATE)
keep_indices = []

for zone in sorted(zone_counts.index):
    idx = np.where(zones == zone)[0]
    n_zone = len(idx)
    if n_zone <= 500:
        n_keep_zone = n_zone
    else:
        n_keep_zone = max(int(n_zone * KEEP_FRAC), 500)
    keep_indices.extend(np.random.choice(idx, size=n_keep_zone, replace=False))

keep_indices = sorted(keep_indices)
X_bal = X[keep_indices]
y_bal = y[keep_indices]

print(f"\n  after balancing: {len(keep_indices)} rows")

print("\n" + "=" * 60)
print("3. Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"  train: {X_train.shape[0]}, test: {X_test.shape[0]}")

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)
y_test_s = scaler_y.transform(y_test)

print("\n" + "=" * 60)
print("4. LightGBM training")
print("=" * 60)

results = {}
for i, tname in enumerate(target_names):
    print(f"\n  --- {tname} ---")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    model.fit(X_train_s, y_train_s[:, i])

    y_pred_s = model.predict(X_test_s)
    y_pred_real = y_pred_s * scaler_y.scale_[i] + scaler_y.mean_[i]
    y_test_real = y_test[:, i]

    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

    baseline_pred = np.full_like(y_test_real, y_train[:, i].mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test_real, baseline_pred))

    results[tname] = {"R2": r2, "MAE": mae, "RMSE": rmse,
                      "Baseline_RMSE": baseline_rmse}

    print(f"    R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

print("\n" + "=" * 60)
print("5. Summary")
print("=" * 60)

print(f"\n{'Target':<12} {'R2':>8} {'MAE':>10} {'RMSE':>10} {'Baseline':>10} {'Improve%':>10}")
print("-" * 60)
for tname, r in results.items():
    imp = (r["Baseline_RMSE"] - r["RMSE"]) / r["Baseline_RMSE"] * 100
    print(f"{tname:<12} {r['R2']:>8.4f} {r['MAE']:>10.4f} "
          f"{r['RMSE']:>10.4f} {r['Baseline_RMSE']:>10.4f} {imp:>9.1f}%")

print("\nDone.")
