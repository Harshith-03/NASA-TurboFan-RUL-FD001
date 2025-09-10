import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skops.io import load as skops_load
from huggingface_hub import hf_hub_download
# ----------------------------
# Paths & constants
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "FD001"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = hf_hub_download(
    repo_id="your-username/nasa-turbofan-rul",  # your repo
    filename="rul_baseline_gbr.skops"
)

model = skops_load(MODEL_PATH, trusted=True)

SCALER_PATH = MODEL_DIR / "scaler.pkl"

ROLL_WINDOWS = (5, 15, 30)   # match your notebook
VAR_THRESH = 1e-8            # match your notebook

COLS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1,22)]

# ----------------------------
# Utility functions
# ----------------------------
def rmse(y_true, y_pred): 
    return mean_squared_error(y_true, y_pred, squared=False)

def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)
    return float(np.sum(s))

def load_fd(path_train, path_test, path_rul):
    for p in (path_train, path_test, path_rul):
        assert Path(p).exists(), f"Missing {p}"
    train = pd.read_csv(path_train, sep=r"\s+", header=None)
    test  = pd.read_csv(path_test,  sep=r"\s+", header=None)
    rul   = pd.read_csv(path_rul,   sep=r"\s+", header=None)
    if train.shape[1] > 26: train = train.iloc[:, :26]
    if test.shape[1]  > 26: test  = test.iloc[:,  :26]
    train.columns = COLS; test.columns = COLS; rul.columns = ["RUL"]
    return train, test, rul

def add_rul_labels(train, test, rul):
    max_cycle_train = train.groupby("unit")["cycle"].max().rename("max_cycle")
    train = train.merge(max_cycle_train, on="unit")
    train["RUL"] = train["max_cycle"] - train["cycle"]
    train.drop(columns=["max_cycle"], inplace=True)

    last_test = test.groupby("unit")["cycle"].max().reset_index(name="last_cycle")
    units_sorted = pd.DataFrame({"unit": sorted(test["unit"].unique())})
    assert len(units_sorted) == len(rul), "Units and RUL length mismatch"
    units_sorted["RUL_at_last"] = rul["RUL"].values
    fail_map = last_test.merge(units_sorted, on="unit")
    fail_map["failure_cycle"] = fail_map["last_cycle"] + fail_map["RUL_at_last"]

    test = test.merge(fail_map[["unit","failure_cycle"]], on="unit", how="left")
    test["true_RUL"] = (test["failure_cycle"] - test["cycle"]).clip(lower=0)
    return train, test

def select_informative_columns(df_train, df_test, threshold=VAR_THRESH):
    sensor_cols = [c for c in df_train.columns if c.startswith("s")]
    setting_cols = ["setting1","setting2"]  # setting3 ~ constant

    # Drop duplicate columns first
    df_train = df_train.loc[:, ~pd.Index(df_train.columns).duplicated()].copy()
    df_test  = df_test.loc[:,  ~pd.Index(df_test.columns).duplicated()].copy()

    var = df_train[sensor_cols].var()
    keep_sensors = var[var > threshold].index.tolist()

    keep = ["unit","cycle"] + setting_cols + keep_sensors

    # make 'keep' unique while preserving order
    seen, keep_unique = set(), []
    for k in keep:
        if k not in seen and k in df_train.columns:
            keep_unique.append(k); seen.add(k)

    return df_train[keep_unique].copy(), df_test[keep_unique].copy(), keep_unique


def add_rolling_stats(df, base_cols, windows=ROLL_WINDOWS):
    df = df.sort_values(["unit","cycle"]).copy()

    # keep only columns that exist and are unique on this frame
    df = df.loc[:, ~pd.Index(df.columns).duplicated()].copy()
    base_cols = [c for c in base_cols if c in df.columns]

    g = df.groupby("unit", group_keys=False)

    for c in base_cols:
        s = df[c]
        # if for any reason a dup sneaks in and s is a DataFrame, take first col
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        df[c] = pd.to_numeric(s, errors="coerce")

    for w in windows:
        for col in base_cols:
            ser = df[col] if not isinstance(df[col], pd.DataFrame) else df[col].iloc[:,0]
            df[f"{col}_mean_{w}"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
            df[f"{col}_std_{w}"]  = g[col].transform(lambda s: s.rolling(w, min_periods=1).std()).fillna(0.0)

    return df


def add_first_diff(df, base_cols):
    df = df.sort_values(["unit","cycle"]).copy()
    g = df.groupby("unit", group_keys=False)
    for c in base_cols:
        if c in df.columns:
            df[f"{c}_diff1"] = g[c].diff().fillna(0.0)
    return df

def build_features(train_lbl, test_lbl):
    train_feats, test_feats, kept_cols = select_informative_columns(train_lbl, test_lbl)
    base_cols = [c for c in kept_cols if c not in ("unit","cycle")]
    train_f = add_rolling_stats(train_feats, base_cols)
    test_f  = add_rolling_stats(test_feats,  base_cols)
    # optional: include first differences if you used them in training
    # train_f = add_first_diff(train_f, base_cols)
    # test_f  = add_first_diff(test_f,  base_cols)

    feat_cols = [c for c in train_f.columns if c not in ("unit","cycle")]
    return train_f, test_f, feat_cols

def last_rows_matrix(df_scaled, label_df, label_col, feat_cols):
    idx = df_scaled.groupby("unit")["cycle"].idxmax()
    X = df_scaled.loc[idx, feat_cols].values
    y = label_df.loc[idx, label_col].values
    units = label_df.loc[idx, "unit"].values
    return X, y, units

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Turbofan RUL Dashboard (FD001)", layout="wide")
st.title("Turbofan RUL — FD001")

# Load data & labels
try:
    train_raw, test_raw, rul = load_fd(
        DATA_DIR / "train_FD001.txt",
        DATA_DIR / "test_FD001.txt",
        DATA_DIR / "RUL_FD001.txt",
    )
except AssertionError as e:
    st.error(str(e))
    st.stop()

train_lbl, test_lbl = add_rul_labels(train_raw.copy(), test_raw.copy(), rul.copy())

# Build features (same pipeline as notebook)
train_f, test_f, feat_cols = build_features(train_lbl, test_lbl)

# Scale with saved scaler
if not SCALER_PATH.exists() or not MODEL_PATH.exists():
    st.warning("Model or scaler not found in /models. Train in the notebook first.")
    st.stop()

scaler = joblib.load(SCALER_PATH)
model  = joblib.load(MODEL_PATH)

train_scaled = train_f.copy()
test_scaled  = test_f.copy()
train_scaled[feat_cols] = scaler.transform(train_f[feat_cols])
test_scaled[feat_cols]  = scaler.transform(test_f[feat_cols])

# Whole-test last-cycle metrics
X_test_last, y_test_last, test_units = last_rows_matrix(test_scaled, test_lbl, "true_RUL", feat_cols)
y_pred_last = model.predict(X_test_last)

c1, c2, c3 = st.columns(3)
c1.metric("RMSE (last cycle)", f"{rmse(y_test_last, y_pred_last):.2f}")
c2.metric("MAE (last cycle)",  f"{mean_absolute_error(y_test_last, y_pred_last):.2f}")
c3.metric("NASA score ↓",      f"{nasa_score(y_test_last, y_pred_last):.1f}")

st.divider()

# Engine picker
all_units = sorted(test_lbl["unit"].unique().tolist())
unit_id = st.selectbox("Select test engine (unit id)", all_units, index=0)

# Build full timeline for selected unit
dfu = test_scaled[test_scaled["unit"] == unit_id].sort_values("cycle").copy()
dfu["pred_RUL"] = model.predict(dfu[feat_cols].values)

truth = test_lbl[test_lbl["unit"] == unit_id].sort_values("cycle")
cycles = truth["cycle"].values
true_rul = truth["true_RUL"].values

# Plot
st.subheader(f"Engine {unit_id}: RUL over time")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(cycles, true_rul, label="True RUL")
ax.plot(cycles, dfu["pred_RUL"].values, label="Predicted RUL")
ax.invert_yaxis()
ax.set_xlabel("Cycle"); ax.set_ylabel("RUL (cycles)")
ax.legend(); ax.grid(True, alpha=0.2)
st.pyplot(fig)

# Last-cycle card
st.caption("Last observed cycle for this engine")
last_row = dfu.iloc[-1]
true_last = true_rul[-1]
pred_last = float(last_row["pred_RUL"])
m1, m2, m3 = st.columns(3)
m1.metric("Last cycle", int(last_row["cycle"]))
m2.metric("True RUL",  f"{true_last:.0f}")
m3.metric("Pred RUL",  f"{pred_last:.0f}")

# Download predictions for this unit
pred_df = pd.DataFrame({"cycle": cycles, "true_RUL": true_rul, "pred_RUL": dfu["pred_RUL"].values})
st.download_button("Download predictions (CSV)", pred_df.to_csv(index=False).encode(), file_name=f"unit_{unit_id}_preds.csv")
