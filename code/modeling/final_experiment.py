import pandas as pd
import numpy as np
import json
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import shap


# =====================================================
# 1. LOAD DATA
# =====================================================

DATA_FILE = "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

df = pd.read_csv(DATA_FILE)

print("Raw dataset:", df.shape)


# =====================================================
# 2. FEATURES
# =====================================================

FEATURES = [
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

TARGET = "Annual_day"


# =====================================================
# 3. FINAL VALIDATION
# =====================================================

df = df.dropna(
    subset=FEATURES + [
        TARGET,
        "Lat",
        "Lon"
    ]
).copy()

df = df[
    df["Population"] >= 0
].copy()

df = df[
    (df["NDVI"] >= -1) &
    (df["NDVI"] <= 1)
].copy()

print("Modeling dataset:", df.shape)


# =====================================================
# 4. SPATIAL BLOCKS
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


# =====================================================
# 5. X AND Y
# =====================================================

X = df[FEATURES]
y = df[TARGET]

groups = df["spatial_block"]


# =====================================================
# 6. FIXED SPATIAL SPLIT
# =====================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_codes = df.iloc[train_idx]["Code"]
test_codes = df.iloc[test_idx]["Code"]

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))
print("Spatial blocks:", groups.nunique())


# =====================================================
# 7. FINAL XGBOOST MODEL
# =====================================================

MODEL_PARAMETERS = {

    "n_estimators": 500,

    "learning_rate": 0.03,

    "max_depth": 6,

    "min_child_weight": 5,

    "subsample": 0.8,

    "colsample_bytree": 0.9,

    "reg_lambda": 1.0,

    "objective": "reg:squarederror",

    "eval_metric": "rmse",

    "random_state": 42,

    "n_jobs": -1,

    "tree_method": "hist"
}


model = XGBRegressor(
    **MODEL_PARAMETERS
)


print("\nTraining final XGBoost...")

model.fit(
    X_train,
    y_train
)


# =====================================================
# 8. PREDICTIONS
# =====================================================

prediction = model.predict(
    X_test
)


# =====================================================
# 9. METRICS
# =====================================================

r2 = r2_score(
    y_test,
    prediction
)

rmse = mean_squared_error(
    y_test,
    prediction
) ** 0.5

mae = mean_absolute_error(
    y_test,
    prediction
)


print("\n======================================")
print("FINAL MODEL PERFORMANCE")
print("======================================")

print("R2   :", round(r2, 6))
print("RMSE :", round(rmse, 6))
print("MAE  :", round(mae, 6))


# =====================================================
# 10. SAVE PREDICTIONS
# =====================================================

predictions = df.iloc[test_idx][
    [
        "Code",
        "Lat",
        "Lon",
        TARGET
    ]
].copy()

predictions[
    "Predicted_Annual_day"
] = prediction

predictions[
    "Residual"
] = (
    predictions[TARGET]
    - predictions["Predicted_Annual_day"]
)

predictions.to_csv(
    "YCEO_UHI_Final_Predictions.csv",
    index=False
)


# =====================================================
# 11. SAVE METRICS
# =====================================================

metrics = pd.DataFrame([
    {
        "Model": "XGBoost",
        "Validation": "Spatial 10-degree blocks",
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Train_samples": len(X_train),
        "Test_samples": len(X_test),
        "Spatial_blocks": groups.nunique()
    }
])

metrics.to_csv(
    "YCEO_UHI_Final_Model_Metrics.csv",
    index=False
)


# =====================================================
# 12. SHAP
# =====================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer(
    X_test
)


# =====================================================
# 13. SHAP IMPORTANCE
# =====================================================

mean_abs_shap = np.abs(
    shap_values.values
).mean(axis=0)

shap_importance = pd.DataFrame({

    "Feature": FEATURES,

    "Mean_Absolute_SHAP":
        mean_abs_shap

})

shap_importance = (
    shap_importance
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
)

shap_importance.to_csv(
    "YCEO_UHI_Final_SHAP_Importance.csv",
    index=False
)


# =====================================================
# 14. SHAP VALUES + PREDICTIONS
# =====================================================

shap_table = pd.DataFrame(
    shap_values.values,
    columns=FEATURES
)

shap_table.insert(
    0,
    "Code",
    test_codes.values
)

shap_table["Actual_Annual_day"] = (
    y_test.values
)

shap_table["Predicted_Annual_day"] = (
    prediction
)

shap_table.to_csv(
    "YCEO_UHI_Final_SHAP_Values.csv",
    index=False
)


# =====================================================
# 15. SAVE TRAIN / TEST IDS
# =====================================================

pd.DataFrame({
    "Code": train_codes.values
}).to_csv(
    "YCEO_UHI_Train_Codes.csv",
    index=False
)

pd.DataFrame({
    "Code": test_codes.values
}).to_csv(
    "YCEO_UHI_Test_Codes.csv",
    index=False
)


# =====================================================
# 16. SAVE MODEL
# =====================================================

joblib.dump(
    model,
    "YCEO_UHI_Final_XGBoost.joblib"
)


# =====================================================
# 17. SAVE EXPERIMENT CONFIG
# =====================================================

config = {

    "dataset": DATA_FILE,

    "features": FEATURES,

    "target": TARGET,

    "spatial_block_size_degrees": 10,

    "test_size": 0.20,

    "random_state": 42,

    "model": "XGBoost",

    "model_parameters": MODEL_PARAMETERS,

    "metrics": {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae)
    }

}

with open(
    "YCEO_UHI_Experiment_Config.json",
    "w"
) as f:

    json.dump(
        config,
        f,
        indent=4
    )


# =====================================================
# COMPLETE
# =====================================================

print("\n======================================")
print("EXPERIMENT LOCKED")
print("======================================")

print("\nFiles created:")

print("YCEO_UHI_Final_Predictions.csv")
print("YCEO_UHI_Final_Model_Metrics.csv")
print("YCEO_UHI_Final_SHAP_Importance.csv")
print("YCEO_UHI_Final_SHAP_Values.csv")
print("YCEO_UHI_Train_Codes.csv")
print("YCEO_UHI_Test_Codes.csv")
print("YCEO_UHI_Final_XGBoost.joblib")
print("YCEO_UHI_Experiment_Config.json")

print("\nFINAL R2:", round(r2, 6))