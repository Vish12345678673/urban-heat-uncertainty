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

# Number of neighbors used to estimate local training density.
K_NEIGHBORS = 5

# Strength of emphasis on rare / difficult training samples.
# 0 = identical to baseline.
# Larger values = stronger tail emphasis.
ALPHA = 1.5

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
# LOAD DATA
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


X = df[FEATURES]
y = df[TARGET]
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
# TRAINING-ONLY DIFFICULTY SCORE
# ============================================================
#
# We estimate how unusual each TRAINING observation is relative
# to the rest of the training distribution.
#
# IMPORTANT:
# This function NEVER uses validation/test observations.
# ============================================================

def calculate_training_difficulty(
    X_train
):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    # Need at least two neighbors because one point should
    # not be its own nearest neighbor.
    k = min(
        K_NEIGHBORS + 1,
        len(X_train)
    )

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        n_jobs=-1
    )

    nn.fit(
        X_scaled
    )

    distances, _ = nn.kneighbors(
        X_scaled
    )

    # Remove self-distance in first column.
    local_distance = distances[:, 1:].mean(
        axis=1
    )

    return local_distance


# ============================================================
# VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)


baseline_results = []
robust_results = []

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


    # ========================================================
    # BASELINE
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
    # TRAINING-ONLY DIFFICULTY
    # ========================================================

    difficulty = calculate_training_difficulty(
        X_train
    )


    # ========================================================
    # CONVERT DIFFICULTY TO WEIGHTS
    # ========================================================
    #
    # Larger local distance = rarer training point.
    #
    # Standardize the difficulty within the training fold.
    # Then exponentiate.
    # ========================================================

    difficulty_z = (
        difficulty
        - difficulty.mean()
    ) / (
        difficulty.std() + 1e-12
    )

    robust_weights = np.exp(
        ALPHA
        * difficulty_z
    )


    # Prevent a few extreme observations from dominating.
    robust_weights = np.clip(
        robust_weights,
        0.25,
        4.0
    )


    # Normalize mean weight to 1.
    robust_weights = (
        robust_weights
        / robust_weights.mean()
    )


    # ========================================================
    # ROBUST XGBOOST
    # ========================================================

    robust = make_model()

    robust.fit(
        X_train,
        y_train,
        sample_weight=robust_weights
    )

    robust_pred = robust.predict(
        X_test
    )


    robust_r2 = r2_score(
        y_test,
        robust_pred
    )

    robust_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            robust_pred
        )
    )

    robust_mae = mean_absolute_error(
        y_test,
        robust_pred
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
    print("SHIFT-ROBUST XGBOOST")

    print(
        "R²   =",
        round(
            robust_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            robust_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            robust_mae,
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


    robust_results.append({

        "Fold": fold,

        "Model": "Shift_Robust_XGBoost",

        "R2": robust_r2,

        "RMSE": robust_rmse,

        "MAE": robust_mae
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

    meta["Baseline_Predicted"] = (
        baseline_pred
    )

    meta["Baseline_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - baseline_pred
        )
    )

    meta["Robust_Predicted"] = (
        robust_pred
    )

    meta["Robust_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - robust_pred
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

robust_df = pd.DataFrame(
    robust_results
)

comparison_df = pd.concat(
    [
        baseline_df,
        robust_df
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
# RECONSTRUCT LEAKAGE-FREE TEST SHIFT
# ============================================================

shift_records = []


for fold in sorted(
    obs_df["Fold"].unique()
):

    split_number = 0

    for current_fold, (
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

        if current_fold == fold:
            break


    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]


    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    test_scaled = scaler.transform(
        X_test
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
        test_scaled
    )

    shift_distance = distances.mean(
        axis=1
    )


    temp = obs_df[
        obs_df["Fold"] == fold
    ].copy()

    temp["Shift_Distance"] = (
        shift_distance
    )

    shift_records.append(
        temp
    )


obs_shift_df = pd.concat(
    shift_records,
    ignore_index=True
)


# ============================================================
# SHIFT QUARTILES
# ============================================================

obs_shift_df["Shift_Quartile"] = pd.qcut(
    obs_shift_df["Shift_Distance"],
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
    obs_shift_df
    .groupby(
        "Shift_Quartile",
        observed=True
    )
    .agg(

        Observations=(
            "Code",
            "count"
        ),

        Mean_Shift_Distance=(
            "Shift_Distance",
            "mean"
        ),

        Baseline_MAE=(
            "Baseline_Absolute_Error",
            "mean"
        ),

        Robust_MAE=(
            "Robust_Absolute_Error",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# DEGRADATION
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


robust_degradation = (
    (
        q4["Robust_MAE"]
        - q1["Robust_MAE"]
    )
    /
    q1["Robust_MAE"]
    * 100
)


# ============================================================
# PRINT RESULTS
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
    "Shift-Robust:",
    round(
        robust_degradation,
        4
    ),
    "%"
)


# ============================================================
# SAVE
# ============================================================

comparison_df.to_csv(
    "YCEO_ShiftRobust_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_ShiftRobust_Model_Summary.csv",
    index=False
)

obs_shift_df.to_csv(
    "YCEO_ShiftRobust_Observation_Results.csv",
    index=False
)

shift_summary.to_csv(
    "YCEO_ShiftRobust_Shift_Quartiles.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("================================================")
print("FILES CREATED")
print("================================================")

print(
    "YCEO_ShiftRobust_Model_Comparison.csv"
)

print(
    "YCEO_ShiftRobust_Model_Summary.csv"
)

print(
    "YCEO_ShiftRobust_Observation_Results.csv"
)

print(
    "YCEO_ShiftRobust_Shift_Quartiles.csv"
)

print()
print(
    "Shift-robust XGBoost experiment complete."
)