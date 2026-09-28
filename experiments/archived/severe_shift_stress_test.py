import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
N_FOLDS = 5

K_NEIGHBORS = 5

# Top 20% = severe environmental shift.
EXTREME_QUANTILE = 0.80

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


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

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


# ============================================================
# SPATIAL BLOCKS
# ============================================================

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


X = df[FEATURES].copy()

y = df[TARGET].copy()

groups = df["spatial_block"].copy()


# ============================================================
# MODEL
# ============================================================

def make_model():

    return XGBRegressor(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=6,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.9,

        reg_lambda=1.0,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )


# ============================================================
# KNN TRAINING-DISTRIBUTION SCORE
# ============================================================

def training_and_test_shift_scores(
    X_train,
    X_test
):

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Training local distance.
    # Self-neighbor is excluded.
    # --------------------------------------------------------

    k_train = min(
        K_NEIGHBORS + 1,
        len(X_train)
    )

    train_nn = NearestNeighbors(
        n_neighbors=k_train,
        metric="euclidean",
        n_jobs=-1
    )

    train_nn.fit(
        train_scaled
    )

    train_distances, _ = (
        train_nn.kneighbors(
            train_scaled
        )
    )

    train_shift = (
        train_distances[:, 1:]
        .mean(
            axis=1
        )
    )


    # --------------------------------------------------------
    # Test distance to outer TRAINING distribution.
    # --------------------------------------------------------

    k_test = min(
        K_NEIGHBORS,
        len(X_train)
    )

    test_nn = NearestNeighbors(
        n_neighbors=k_test,
        metric="euclidean",
        n_jobs=-1
    )

    test_nn.fit(
        train_scaled
    )

    test_distances, _ = (
        test_nn.kneighbors(
            test_scaled
        )
    )

    test_shift = (
        test_distances
        .mean(
            axis=1
        )
    )


    return (
        train_shift,
        test_shift
    )


# ============================================================
# OUTER SPATIAL CV
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)


fold_results = []

observation_results = []


for fold, (
    train_idx,
    test_idx
) in enumerate(

    cv.split(
        X,
        y,
        groups=groups
    ),

    start=1

):

    print()
    print("======================================================")
    print("OUTER SPATIAL FOLD", fold)
    print("======================================================")


    X_train = X.iloc[
        train_idx
    ].copy()

    y_train = y.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    # ========================================================
    # SHIFT SCORES
    # ========================================================

    train_shift, test_shift = (
        training_and_test_shift_scores(
            X_train,
            X_test
        )
    )


    # ========================================================
    # EXTREME THRESHOLD
    #
    # Threshold comes ONLY from the outer training distribution.
    # ========================================================

    threshold = np.quantile(
        train_shift,
        EXTREME_QUANTILE
    )


    # Training observations considered "common".
    common_train_mask = (
        train_shift
        < threshold
    )


    # Outer test observations considered severely shifted.
    extreme_test_mask = (
        test_shift
        >= threshold
    )


    X_common = X_train.loc[
        common_train_mask
    ].copy()

    y_common = y_train.loc[
        common_train_mask
    ].copy()


    X_extreme_test = X_test.loc[
        extreme_test_mask
    ].copy()

    y_extreme_test = y_test.loc[
        extreme_test_mask
    ].copy()


    # ========================================================
    # CHECK
    # ========================================================

    print(
        "Outer training observations:",
        len(X_train)
    )

    print(
        "Common training observations:",
        len(X_common)
    )

    print(
        "Extreme outer-test observations:",
        len(X_extreme_test)
    )


    if len(X_extreme_test) < 10:

        print(
            "Too few extreme test observations."
        )

        continue


    # ========================================================
    # MODEL A
    #
    # STANDARD MODEL
    # ALL OUTER TRAINING DATA
    # ========================================================

    standard_model = make_model()

    standard_model.fit(
        X_train,
        y_train
    )

    standard_pred = (
        standard_model.predict(
            X_extreme_test
        )
    )


    # ========================================================
    # MODEL B
    #
    # SEVERE-SHIFT TRAINING MODEL
    #
    # Train only on the common 80% of the outer training
    # predictor distribution.
    # ========================================================

    shifted_model = make_model()

    shifted_model.fit(
        X_common,
        y_common
    )

    shifted_pred = (
        shifted_model.predict(
            X_extreme_test
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    standard_r2 = r2_score(
        y_extreme_test,
        standard_pred
    )

    standard_rmse = np.sqrt(
        mean_squared_error(
            y_extreme_test,
            standard_pred
        )
    )

    standard_mae = mean_absolute_error(
        y_extreme_test,
        standard_pred
    )


    shifted_r2 = r2_score(
        y_extreme_test,
        shifted_pred
    )

    shifted_rmse = np.sqrt(
        mean_squared_error(
            y_extreme_test,
            shifted_pred
        )
    )

    shifted_mae = mean_absolute_error(
        y_extreme_test,
        shifted_pred
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("STANDARD XGBOOST")
    print(
        "R²   =",
        round(
            standard_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            standard_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            standard_mae,
            6
        )
    )


    print()
    print("SEVERE-SHIFT TRAINED XGBOOST")
    print(
        "R²   =",
        round(
            shifted_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            shifted_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            shifted_mae,
            6
        )
    )


    # ========================================================
    # DELTAS
    # ========================================================

    print()
    print("CHANGES ON EXTREME TEST CASES")

    print(
        "ΔR²   =",
        round(
            shifted_r2
            - standard_r2,
            6
        )
    )

    print(
        "ΔRMSE =",
        round(
            shifted_rmse
            - standard_rmse,
            6
        )
    )

    print(
        "ΔMAE  =",
        round(
            shifted_mae
            - standard_mae,
            6
        )
    )


    # ========================================================
    # STORE FOLD RESULTS
    # ========================================================

    fold_results.append({

        "Fold": fold,

        "Outer_Train_Observations":
            len(X_train),

        "Common_Train_Observations":
            len(X_common),

        "Extreme_Test_Observations":
            len(X_extreme_test),

        "Shift_Threshold":
            threshold,

        "Standard_R2":
            standard_r2,

        "Standard_RMSE":
            standard_rmse,

        "Standard_MAE":
            standard_mae,

        "SevereShift_R2":
            shifted_r2,

        "SevereShift_RMSE":
            shifted_rmse,

        "SevereShift_MAE":
            shifted_mae,

        "Delta_R2":
            shifted_r2
            - standard_r2,

        "Delta_RMSE":
            shifted_rmse
            - standard_rmse,

        "Delta_MAE":
            shifted_mae
            - standard_mae
    })


    # ========================================================
    # OBSERVATION RESULTS
    # ========================================================

    meta = df.iloc[
        test_idx
    ][
        [
            "Code",
            "Lat",
            "Lon",
            "spatial_block"
        ]
    ].copy()


    meta = meta.loc[
        extreme_test_mask
    ].copy()


    meta["Fold"] = fold

    meta["Test_Shift_Score"] = (
        test_shift[
            extreme_test_mask
        ]
    )

    meta["Observed"] = (
        y_extreme_test.to_numpy()
    )

    meta["Standard_Predicted"] = (
        standard_pred
    )

    meta["Standard_Absolute_Error"] = (
        np.abs(
            y_extreme_test.to_numpy()
            -
            standard_pred
        )
    )

    meta["SevereShift_Predicted"] = (
        shifted_pred
    )

    meta["SevereShift_Absolute_Error"] = (
        np.abs(
            y_extreme_test.to_numpy()
            -
            shifted_pred
        )
    )


    observation_results.append(
        meta
    )


# ============================================================
# RESULTS
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# OVERALL SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "R2",
        "RMSE",
        "MAE"
    ],

    "Standard_Mean": [

        fold_df["Standard_R2"].mean(),

        fold_df["Standard_RMSE"].mean(),

        fold_df["Standard_MAE"].mean()
    ],

    "SevereShift_Mean": [

        fold_df["SevereShift_R2"].mean(),

        fold_df["SevereShift_RMSE"].mean(),

        fold_df["SevereShift_MAE"].mean()
    ],

    "Standard_SD": [

        fold_df["Standard_R2"].std(
            ddof=1
        ),

        fold_df["Standard_RMSE"].std(
            ddof=1
        ),

        fold_df["Standard_MAE"].std(
            ddof=1
        )
    ],

    "SevereShift_SD": [

        fold_df["SevereShift_R2"].std(
            ddof=1
        ),

        fold_df["SevereShift_RMSE"].std(
            ddof=1
        ),

        fold_df["SevereShift_MAE"].std(
            ddof=1
        )
    ]
})


# ============================================================
# POOLED EXTREME-CASE PERFORMANCE
# ============================================================

pooled_standard_mae = (
    obs_df[
        "Standard_Absolute_Error"
    ].mean()
)

pooled_severe_mae = (
    obs_df[
        "SevereShift_Absolute_Error"
    ].mean()
)


print()
print("======================================================")
print("SEVERE-SHIFT SUMMARY")
print("======================================================")


print(
    summary.to_string(
        index=False
    )
)


print()
print(
    "Pooled extreme-test Standard MAE:",
    round(
        pooled_standard_mae,
        6
    )
)

print(
    "Pooled extreme-test SevereShift MAE:",
    round(
        pooled_severe_mae,
        6
    )
)


print()
print("======================================================")
print("FOLD RESULTS")
print("======================================================")

print(
    fold_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_SevereShift_FoldResults.csv",
    index=False
)

summary.to_csv(
    "YCEO_SevereShift_Summary.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_SevereShift_ExtremeTest_Observations.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("======================================================")
print("FILES CREATED")
print("======================================================")

print(
    "YCEO_SevereShift_FoldResults.csv"
)

print(
    "YCEO_SevereShift_Summary.csv"
)

print(
    "YCEO_SevereShift_ExtremeTest_Observations.csv"
)

print()
print(
    "Severe distribution-shift stress test complete."
)