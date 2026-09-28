import pandas as pd
import numpy as np
import time

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
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
# 3. CLEAN DATA
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
# 4. CREATE 10° SPATIAL BLOCKS
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
# 6. GROUP K-FOLD
# =====================================================

cv = GroupKFold(
    n_splits=5
)


# =====================================================
# 7. RESULTS
# =====================================================

results = []


# =====================================================
# 8. RUN FIVE SPATIAL FOLDS
# =====================================================

for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print()
    print(
        "======================================"
    )

    print(
        "SPATIAL FOLD",
        fold
    )

    print(
        "======================================"
    )


    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]


    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )


    # -------------------------------------------------
    # XGBOOST
    # -------------------------------------------------

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


    start_time = time.time()


    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    elapsed = (
        time.time()
        - start_time
    )


    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------

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
        round(
            r2,
            6
        )
    )

    print(
        "RMSE :",
        round(
            rmse,
            6
        )
    )

    print(
        "MAE  :",
        round(
            mae,
            6
        )
    )


    results.append({

        "Fold": fold,

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae,

        "Train_samples":
            len(X_train),

        "Test_samples":
            len(X_test),

        "Training_time_seconds":
            elapsed
    })


# =====================================================
# 9. RESULTS DATAFRAME
# =====================================================

results_df = pd.DataFrame(
    results
)


# =====================================================
# 10. MEAN AND STANDARD DEVIATION
# =====================================================

summary = pd.DataFrame({

    "Metric": [
        "R2",
        "RMSE",
        "MAE"
    ],

    "Mean": [
        results_df["R2"].mean(),
        results_df["RMSE"].mean(),
        results_df["MAE"].mean()
    ],

    "SD": [
        results_df["R2"].std(ddof=1),
        results_df["RMSE"].std(ddof=1),
        results_df["MAE"].std(ddof=1)
    ]
})


# =====================================================
# 11. SAVE
# =====================================================

results_df.to_csv(
    "YCEO_UHI_5Fold_Spatial_CV.csv",
    index=False
)

summary.to_csv(
    "YCEO_UHI_5Fold_Spatial_CV_Summary.csv",
    index=False
)


# =====================================================
# 12. PRINT FINAL RESULTS
# =====================================================

print()
print()
print(
    "======================================"
)

print(
    "5-FOLD SPATIAL CROSS-VALIDATION"
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
    "======================================"
)

print(
    "MEAN ± SD"
)

print(
    "======================================"
)

print()

for _, row in summary.iterrows():

    print(
        row["Metric"],
        "=",
        round(
            row["Mean"],
            6
        ),
        "±",
        round(
            row["SD"],
            6
        )
    )


print()
print(
    "Saved:"
)

print(
    "YCEO_UHI_5Fold_Spatial_CV.csv"
)

print(
    "YCEO_UHI_5Fold_Spatial_CV_Summary.csv"
)

print()
print(
    "5-fold spatial validation complete."
)