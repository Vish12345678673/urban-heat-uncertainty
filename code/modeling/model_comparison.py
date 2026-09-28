import pandas as pd
import numpy as np
import time

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from xgboost import XGBRegressor


# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv(
    "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"
)

print("Dataset:", df.shape)


# =====================================================
# 2. FEATURES
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
# 3. CLEANING
# =====================================================

df = df.dropna(
    subset=features + [
        target,
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
# 5. DATA
# =====================================================

X = df[features]

y = df[target]

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


print()
print("Train:", len(X_train))
print("Test :", len(X_test))
print(
    "Spatial blocks:",
    groups.nunique()
)


# =====================================================
# 7. MODELS
# =====================================================

models = {

    "Linear Regression":

        LinearRegression(),


    "Random Forest":

        RandomForestRegressor(

            n_estimators=300,

            max_depth=None,

            min_samples_leaf=2,

            max_features=0.8,

            random_state=42,

            n_jobs=-1
        ),


    "HistGradientBoosting":

        HistGradientBoostingRegressor(

            learning_rate=0.03,

            max_iter=500,

            max_leaf_nodes=31,

            min_samples_leaf=30,

            l2_regularization=1.0,

            random_state=42
        ),


    "XGBoost":

        XGBRegressor(

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
}


# =====================================================
# 8. TRAIN AND EVALUATE
# =====================================================

results = []


for name, model in models.items():

    print()
    print(
        "======================================"
    )

    print(
        "Training:",
        name
    )

    print(
        "======================================"
    )

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    elapsed = (
        time.time()
        - start
    )


    r2 = r2_score(
        y_test,
        prediction
    )

    rmse = (
        mean_squared_error(
            y_test,
            prediction
        )
        ** 0.5
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )


    print(
        "R²   :",
        round(r2, 6)
    )

    print(
        "RMSE :",
        round(rmse, 6)
    )

    print(
        "MAE  :",
        round(mae, 6)
    )

    print(
        "Time :",
        round(elapsed, 2),
        "seconds"
    )


    results.append({

        "Model": name,

        "Spatial_R2": r2,

        "Spatial_RMSE": rmse,

        "Spatial_MAE": mae,

        "Training_Time_seconds":
            elapsed
    })


# =====================================================
# 9. RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Spatial_R2",
    ascending=False
).reset_index(
    drop=True
)


# =====================================================
# 10. SAVE
# =====================================================

results_df.to_csv(
    "YCEO_UHI_Model_Comparison.csv",
    index=False
)


# =====================================================
# 11. PRINT FINAL TABLE
# =====================================================

print()
print()
print(
    "======================================"
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "======================================"
)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()

print(
    "Best model:",
    results_df.iloc[0]["Model"]
)

print(
    "Best spatial R²:",
    round(
        results_df.iloc[0]["Spatial_R2"],
        6
    )
)

print()

print(
    "Saved:"
)

print(
    "YCEO_UHI_Model_Comparison.csv"
)

print()

print(
    "Model comparison complete."
)