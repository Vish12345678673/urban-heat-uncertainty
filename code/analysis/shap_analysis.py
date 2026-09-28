import pandas as pd
import numpy as np
import subprocess
import sys
import importlib.util

# Install SHAP if necessary
if importlib.util.find_spec("shap") is None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "shap", "-q"
    ])

import shap

from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"
)


# =====================================================
# FEATURES
# =====================================================

features = [
    "NDVI",
    "Elevation",
    "Builtup",
    "Population",
    "Mean_Temp_C",
    "Mean_Monthly_Precip_mm",
    "Urban_LC_Fraction",
    "Water_LC_Fraction",
    "Vegetation_LC_Fraction"
]

target = "Annual_day"


# =====================================================
# CLEAN
# =====================================================

df = df.dropna(
    subset=features + [
        target,
        "Lat",
        "Lon"
    ]
).copy()


# =====================================================
# SPATIAL BLOCKS
# =====================================================

df["lat_block"] = np.floor(
    df["Lat"] / 10
).astype(int)

df["lon_block"] = np.floor(
    df["Lon"] / 10
).astype(int)

df["spatial_block"] = (
    df["lat_block"].astype(str)
    + "_"
    + df["lon_block"].astype(str)
)


X = df[features]
y = df[target]

groups = df["spatial_block"]


# =====================================================
# SAME SPATIAL SPLIT
# =====================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# =====================================================
# XGBOOST
# =====================================================

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)


print("Training XGBoost...")

model.fit(
    X_train,
    y_train
)


prediction = model.predict(
    X_test
)

print(
    "Spatial R2:",
    round(
        r2_score(
            y_test,
            prediction
        ),
        4
    )
)


# =====================================================
# SHAP
# =====================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer(
    X_test
)


# =====================================================
# GLOBAL SHAP IMPORTANCE
# =====================================================

mean_abs_shap = np.abs(
    shap_values.values
).mean(axis=0)

importance = pd.DataFrame({

    "Feature": features,

    "Mean_Absolute_SHAP":
        mean_abs_shap

})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


importance.to_csv(
    "YCEO_UHI_SHAP_Importance.csv",
    index=False
)


# =====================================================
# SAVE SHAP VALUES
# =====================================================

shap_df = pd.DataFrame(
    shap_values.values,
    columns=features
)

shap_df["Annual_day_actual"] = (
    y_test.values
)

shap_df["Annual_day_predicted"] = (
    prediction
)

shap_df.to_csv(
    "YCEO_UHI_SHAP_Values.csv",
    index=False
)


# =====================================================
# PRINT RESULTS
# =====================================================

print("\n===================================")
print("SHAP FEATURE IMPORTANCE")
print("===================================")

print(
    importance.to_string(
        index=False
    )
)


print("\nFiles created:")

print(
    "YCEO_UHI_SHAP_Importance.csv"
)

print(
    "YCEO_UHI_SHAP_Values.csv"
)

print("\nSHAP analysis complete.")