import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("YCEO_UHI_Enhanced_ML_Clean_Daytime.csv")


# =====================================================
# FEATURES AND TARGET
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
# REMOVE MISSING VALUES
# =====================================================

df = df.dropna(
    subset=features + [target, "Lat", "Lon"]
).copy()


# =====================================================
# CREATE 10° SPATIAL BLOCKS
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
# DATA
# =====================================================

X = df[features]
y = df[target]
groups = df["spatial_block"]


# =====================================================
# SPATIAL TRAIN / TEST SPLIT
# =====================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Spatial blocks:", groups.nunique())


# =====================================================
# HYPERPARAMETER CONFIGURATIONS
# =====================================================

configs = [

    {
        "learning_rate": 0.03,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.5
    },

    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.5
    },

    {
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.08,
        "max_iter": 300,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0
    },

    {
        "learning_rate": 0.03,
        "max_iter": 500,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0
    }
]


# =====================================================
# RUN EXPERIMENT
# =====================================================

results = []

best_r2 = -np.inf
best_config = None


for i, config in enumerate(configs, start=1):

    print("\nRunning configuration", i)

    model = HistGradientBoostingRegressor(
        random_state=42,
        **config
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

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

    print(
        "R2:",
        round(r2, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "MAE:",
        round(mae, 4)
    )

    results.append({

        "Configuration": i,

        **config,

        "Spatial_R2": r2,

        "Spatial_RMSE": rmse,

        "Spatial_MAE": mae

    })


    if r2 > best_r2:

        best_r2 = r2

        best_config = config


# =====================================================
# SAVE RESULTS
# =====================================================

results = pd.DataFrame(
    results
)

results = results.sort_values(
    "Spatial_R2",
    ascending=False
)

results.to_csv(
    "YCEO_UHI_HGB_Tuning_Results.csv",
    index=False
)


# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n===================================")
print("BEST CONFIGURATION")
print("===================================")

print(best_config)

print(
    "\nBEST SPATIAL R2:",
    round(best_r2, 4)
)

print("\n===================================")
print("ALL RESULTS")
print("===================================")

print(
    results.to_string(
        index=False
    )
)

print(
    "\nResults saved to:"
    " YCEO_UHI_HGB_Tuning_Results.csv"
)