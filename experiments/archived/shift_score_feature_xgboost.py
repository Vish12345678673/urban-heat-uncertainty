import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
N_FOLDS = 5
K_NEIGHBORS = 5

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
# LOAD AND CLEAN
# ============================================================

df = pd.read_csv(DATA_FILE)

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
groups = df["spatial_block"]


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
# LEAKAGE-FREE SHIFT SCORE
# ============================================================

def compute_shift_score(
    X_train,
    X_target
):

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    target_scaled = scaler.transform(
        X_target
    )

    k = min(
        K_NEIGHBORS,
        len(X_train)
    )

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        n_jobs=-1
    )

    nn.fit(
        train_scaled
    )

    distances, _ = nn.kneighbors(
        target_scaled
    )

    shift_score = distances.mean(
        axis=1
    )

    return shift_score


# ============================================================
# 5-FOLD SPATIAL CV
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)

baseline_results = []
shift_feature_results = []

observation_results = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print()
    print("================================================")
    print("SPATIAL FOLD", fold)
    print("================================================")


    X_train = X.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_train = y.iloc[
        train_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    # ========================================================
    # BASELINE XGBOOST
    # ========================================================

    baseline = make_model()

    baseline.fit(
        X_train,
        y_train
    )

    baseline_pred = baseline.predict(
        X_test
    )


    baseline_r2 = r2_score(
        y_test,
        baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_pred
        )
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )


    # ========================================================
    # COMPUTE SHIFT SCORES
    # ========================================================
    #
    # Test shift score:
    #   distance from the training distribution.
    #
    # Training shift scores:
    #   leave-one-out style local distances within TRAIN ONLY.
    #
    # No validation observation is used to construct the
    # training shift feature.
    # ========================================================

    train_scaler = StandardScaler()

    train_scaled = train_scaler.fit_transform(
        X_train
    )

    test_scaled = train_scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Training local-density score
    # --------------------------------------------------------

    k_train = min(
        K_NEIGHBORS + 1,
        len(X_train)
    )

    nn_train = NearestNeighbors(
        n_neighbors=k_train,
        metric="euclidean",
        n_jobs=-1
    )

    nn_train.fit(
        train_scaled
    )

    train_distances, _ = (
        nn_train.kneighbors(
            train_scaled
        )
    )


    # First neighbor is itself.
    train_shift = (
        train_distances[:, 1:].mean(
            axis=1
        )
    )


    # --------------------------------------------------------
    # Test score relative to TRAIN distribution
    # --------------------------------------------------------

    k_test = min(
        K_NEIGHBORS,
        len(X_train)
    )

    nn_test = NearestNeighbors(
        n_neighbors=k_test,
        metric="euclidean",
        n_jobs=-1
    )

    nn_test.fit(
        train_scaled
    )

    test_distances, _ = (
        nn_test.kneighbors(
            test_scaled
        )
    )

    test_shift = (
        test_distances.mean(
            axis=1
        )
    )


    # ========================================================
    # STANDARDIZE SHIFT FEATURE
    # ========================================================
    #
    # Training statistics ONLY.
    # ========================================================

    train_mean = train_shift.mean()
    train_std = (
        train_shift.std()
        + 1e-12
    )

    train_shift_z = (
        train_shift
        - train_mean
    ) / train_std

    test_shift_z = (
        test_shift
        - train_mean
    ) / train_std


    # ========================================================
    # AUGMENT FEATURES
    # ========================================================

    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()

    X_train_aug[
        "Distribution_Shift_Score"
    ] = train_shift_z

    X_test_aug[
        "Distribution_Shift_Score"
    ] = test_shift_z


    # ========================================================
    # TRAIN SHIFT-AWARE MODEL
    # ========================================================

    shift_model = make_model()

    shift_model.fit(
        X_train_aug,
        y_train
    )

    shift_pred = shift_model.predict(
        X_test_aug
    )


    shift_r2 = r2_score(
        y_test,
        shift_pred
    )

    shift_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            shift_pred
        )
    )

    shift_mae = mean_absolute_error(
        y_test,
        shift_pred
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("BASELINE XGBOOST")

    print(
        "R²   =",
        round(
            baseline_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            baseline_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            baseline_mae,
            6
        )
    )


    print()
    print("SHIFT-SCORE XGBOOST")

    print(
        "R²   =",
        round(
            shift_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            shift_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            shift_mae,
            6
        )
    )


    # ========================================================
    # STORE FOLD RESULTS
    # ========================================================

    baseline_results.append({

        "Fold": fold,

        "Model": "Baseline_XGBoost",

        "R2": baseline_r2,

        "RMSE": baseline_rmse,

        "MAE": baseline_mae
    })


    shift_feature_results.append({

        "Fold": fold,

        "Model": "Shift_Score_XGBoost",

        "R2": shift_r2,

        "RMSE": shift_rmse,

        "MAE": shift_mae
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

    meta["Fold"] = fold

    meta["Observed"] = (
        y_test.to_numpy()
    )

    meta["Shift_Score"] = (
        test_shift_z
    )

    meta["Baseline_Predicted"] = (
        baseline_pred
    )

    meta["Baseline_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - baseline_pred
        )
    )

    meta["ShiftAware_Predicted"] = (
        shift_pred
    )

    meta["ShiftAware_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - shift_pred
        )
    )

    observation_results.append(
        meta
    )


# ============================================================
# COMBINE
# ============================================================

baseline_df = pd.DataFrame(
    baseline_results
)

shift_df = pd.DataFrame(
    shift_feature_results
)

comparison_df = pd.concat(
    [
        baseline_df,
        shift_df
    ],
    ignore_index=True
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    comparison_df
    .groupby("Model")
    .agg(

        Mean_R2=(
            "R2",
            "mean"
        ),

        SD_R2=(
            "R2",
            "std"
        ),

        Mean_RMSE=(
            "RMSE",
            "mean"
        ),

        SD_RMSE=(
            "RMSE",
            "std"
        ),

        Mean_MAE=(
            "MAE",
            "mean"
        ),

        SD_MAE=(
            "MAE",
            "std"
        )
    )
    .reset_index()
)


# ============================================================
# SHIFT QUARTILES
# ============================================================

obs_df["Shift_Quartile"] = pd.qcut(
    obs_df["Shift_Score"],
    q=4,
    labels=[
        "Q1 - closest",
        "Q2",
        "Q3",
        "Q4 - farthest"
    ],
    duplicates="drop"
)


shift_summary = (
    obs_df
    .groupby(
        "Shift_Quartile",
        observed=True
    )
    .agg(

        Observations=(
            "Code",
            "count"
        ),

        Mean_Shift_Score=(
            "Shift_Score",
            "mean"
        ),

        Baseline_MAE=(
            "Baseline_Absolute_Error",
            "mean"
        ),

        ShiftAware_MAE=(
            "ShiftAware_Absolute_Error",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# Q1-Q4 DEGRADATION
# ============================================================

q1 = shift_summary[
    shift_summary["Shift_Quartile"]
    == "Q1 - closest"
].iloc[0]

q4 = shift_summary[
    shift_summary["Shift_Quartile"]
    == "Q4 - farthest"
].iloc[0]


baseline_degradation = (
    (
        q4["Baseline_MAE"]
        - q1["Baseline_MAE"]
    )
    /
    q1["Baseline_MAE"]
    * 100
)

shift_degradation = (
    (
        q4["ShiftAware_MAE"]
        - q1["ShiftAware_MAE"]
    )
    /
    q1["ShiftAware_MAE"]
    * 100
)


# ============================================================
# PRINT FINAL
# ============================================================

print()
print("================================================")
print("MODEL SUMMARY")
print("================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("================================================")
print("SHIFT-QUARTILE PERFORMANCE")
print("================================================")

print(
    shift_summary.to_string(
        index=False
    )
)


print()
print("================================================")
print("Q4 VS Q1 DEGRADATION")
print("================================================")

print(
    "Baseline:",
    round(
        baseline_degradation,
        4
    ),
    "%"
)

print(
    "Shift-aware:",
    round(
        shift_degradation,
        4
    ),
    "%"
)


# ============================================================
# SAVE
# ============================================================

comparison_df.to_csv(
    "YCEO_ShiftScore_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_ShiftScore_Model_Summary.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_ShiftScore_Observation_Results.csv",
    index=False
)

shift_summary.to_csv(
    "YCEO_ShiftScore_Shift_Quartiles.csv",
    index=False
)


print()
print("================================================")
print("FILES CREATED")
print("================================================")

print(
    "YCEO_ShiftScore_Model_Comparison.csv"
)

print(
    "YCEO_ShiftScore_Model_Summary.csv"
)

print(
    "YCEO_ShiftScore_Observation_Results.csv"
)

print(
    "YCEO_ShiftScore_Shift_Quartiles.csv"
)

print()
print(
    "Shift-score feature experiment complete."
)