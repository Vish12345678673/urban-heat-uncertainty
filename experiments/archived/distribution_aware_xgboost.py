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

# Number of target neighbors used to measure how close each
# training observation is to the target/test distribution.
K_TARGET = 5

# Controls how strongly similarity affects training weights.
# Larger alpha = stronger adaptation.
ALPHA = 2.0

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


X = df[FEATURES]
y = df[TARGET]
groups = df["spatial_block"]


# ============================================================
# MODEL FACTORY
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
# HELPER: CREATE DISTRIBUTION-AWARE WEIGHTS
# ============================================================

def calculate_target_similarity_weights(
    X_train,
    X_target
):

    # --------------------------------------------------------
    # Standardize using TRAINING DATA ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    target_scaled = scaler.transform(
        X_target
    )


    # --------------------------------------------------------
    # Find nearest TARGET observations for each training
    # observation.
    #
    # Small distance = training observation looks like
    # the target/test distribution.
    # --------------------------------------------------------

    k = min(
        K_TARGET,
        len(X_target)
    )

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        n_jobs=-1
    )

    nn.fit(
        target_scaled
    )


    distances, _ = nn.kneighbors(
        train_scaled
    )


    mean_distance = (
        distances.mean(
            axis=1
        )
    )


    # --------------------------------------------------------
    # Convert similarity into positive weights.
    #
    # Similar observations receive larger weights.
    # Distant observations receive smaller weights.
    #
    # exp(-alpha*d) keeps weights positive.
    # --------------------------------------------------------

    weights = np.exp(
        -ALPHA
        * mean_distance
    )


    # --------------------------------------------------------
    # Normalize mean weight to 1.
    # This keeps the overall scale of the objective
    # comparable between folds.
    # --------------------------------------------------------

    weights = (
        weights
        / weights.mean()
    )


    return weights, mean_distance


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)


baseline_results = []
adapted_results = []

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
    # DISTRIBUTION-AWARE WEIGHTS
    # ========================================================

    weights, train_target_distance = (
        calculate_target_similarity_weights(
            X_train,
            X_test
        )
    )


    # ========================================================
    # DISTRIBUTION-AWARE XGBOOST
    # ========================================================

    adapted = make_model()

    adapted.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    adapted_pred = adapted.predict(
        X_test
    )


    adapted_r2 = r2_score(
        y_test,
        adapted_pred
    )

    adapted_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            adapted_pred
        )
    )

    adapted_mae = mean_absolute_error(
        y_test,
        adapted_pred
    )


    # ========================================================
    # PRINT FOLD RESULTS
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
    print("DISTRIBUTION-AWARE XGBOOST")

    print(
        "R²   =",
        round(
            adapted_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            adapted_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            adapted_mae,
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


    adapted_results.append({

        "Fold": fold,

        "Model": "Distribution_Aware_XGBoost",

        "R2": adapted_r2,

        "RMSE": adapted_rmse,

        "MAE": adapted_mae
    })


    # ========================================================
    # OBSERVATION-LEVEL RESULTS
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

    meta["Adapted_Predicted"] = (
        adapted_pred
    )

    meta["Adapted_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - adapted_pred
        )
    )

    observation_results.append(
        meta
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

baseline_df = pd.DataFrame(
    baseline_results
)

adapted_df = pd.DataFrame(
    adapted_results
)

comparison_df = pd.concat(
    [
        baseline_df,
        adapted_df
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
# SHIFT-SPECIFIC PERFORMANCE
# ============================================================
#
# Calculate shift distance for the TEST observations using
# only the corresponding training distribution.
#
# This is the same leakage-free shift definition used in V2.
# ============================================================

shift_records = []


for fold in sorted(
    obs_df["Fold"].unique()
):

    train_idx, test_idx = next(
        (
            (tr, te)
            for f, (tr, te)
            in enumerate(
                cv.split(
                    X,
                    y,
                    groups=groups
                ),
                start=1
            )
            if f == fold
        )
    )


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
        K_TARGET,
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


    shift = distances.mean(
        axis=1
    )


    temp = obs_df[
        obs_df["Fold"] == fold
    ].copy()

    temp["Shift_Distance"] = shift

    shift_records.append(
        temp
    )


obs_shift_df = pd.concat(
    shift_records,
    ignore_index=True
)


# ============================================================
# Q1-Q4 PERFORMANCE
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

        Adapted_MAE=(
            "Adapted_Absolute_Error",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# SHIFT DEGRADATION
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
    / q1["Baseline_MAE"]
    * 100
)

adapted_degradation = (
    (
        q4["Adapted_MAE"]
        - q1["Adapted_MAE"]
    )
    / q1["Adapted_MAE"]
    * 100
)


# ============================================================
# PRINT SUMMARY
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
    "Baseline degradation:",
    round(
        baseline_degradation,
        4
    ),
    "%"
)

print(
    "Adapted degradation:",
    round(
        adapted_degradation,
        4
    ),
    "%"
)


# ============================================================
# SAVE
# ============================================================

comparison_df.to_csv(
    "YCEO_DistributionAware_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_DistributionAware_Model_Summary.csv",
    index=False
)

obs_shift_df.to_csv(
    "YCEO_DistributionAware_Observation_Results.csv",
    index=False
)

shift_summary.to_csv(
    "YCEO_DistributionAware_Shift_Quartiles.csv",
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
    "YCEO_DistributionAware_Model_Comparison.csv"
)

print(
    "YCEO_DistributionAware_Model_Summary.csv"
)

print(
    "YCEO_DistributionAware_Observation_Results.csv"
)

print(
    "YCEO_DistributionAware_Shift_Quartiles.csv"
)

print()
print(
    "Distribution-aware baseline complete."
)