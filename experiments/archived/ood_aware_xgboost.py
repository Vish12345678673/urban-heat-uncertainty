import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
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

# Controls emphasis on high-OOD training observations.
# We test one conservative value first.
LAMBDA_OOD = 1.0


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
# OOD SCORE FROM TRAINING DATA ONLY
# ============================================================

def compute_train_ood_weights(X_train):
    """
    Fit the OOD detector on training data only.

    Higher anomaly score = more unusual training observation.

    The resulting weight is:
        1 + lambda * normalized_ood_score

    This creates a conservative robustness objective.
    """

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    detector.fit(
        train_scaled
    )

    # Higher = more unusual.
    raw_ood = -detector.decision_function(
        train_scaled
    )

    # Robust normalization using percentiles.
    p5 = np.percentile(
        raw_ood,
        5
    )

    p95 = np.percentile(
        raw_ood,
        95
    )

    denom = (
        p95 - p5
    )

    if denom <= 1e-12:
        normalized = np.zeros_like(
            raw_ood
        )
    else:
        normalized = (
            raw_ood - p5
        ) / denom

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    weights = (
        1.0
        + LAMBDA_OOD * normalized
    )

    # Keep mean weight exactly 1.
    weights = (
        weights
        / weights.mean()
    )

    return weights


# ============================================================
# CV
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)

baseline_results = []
ood_results = []

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
    # OOD-AWARE TRAINING
    # ========================================================

    ood_weights = compute_train_ood_weights(
        X_train
    )

    ood_model = make_model()

    ood_model.fit(
        X_train,
        y_train,
        sample_weight=ood_weights
    )

    ood_pred = ood_model.predict(
        X_test
    )


    ood_r2 = r2_score(
        y_test,
        ood_pred
    )

    ood_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            ood_pred
        )
    )

    ood_mae = mean_absolute_error(
        y_test,
        ood_pred
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
    print("OOD-AWARE XGBOOST")

    print(
        "R²   =",
        round(
            ood_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            ood_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            ood_mae,
            6
        )
    )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    baseline_results.append({
        "Fold": fold,
        "Model": "Baseline_XGBoost",
        "R2": baseline_r2,
        "RMSE": baseline_rmse,
        "MAE": baseline_mae
    })

    ood_results.append({
        "Fold": fold,
        "Model": "OOD_Aware_XGBoost",
        "R2": ood_r2,
        "RMSE": ood_rmse,
        "MAE": ood_mae
    })


    # ========================================================
    # TEST OOD SCORE FOR EVALUATION
    # ========================================================
    #
    # IMPORTANT:
    # Test OOD is computed relative to the training distribution.
    # Test labels are never used.
    # ========================================================

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    test_scaled = scaler.transform(
        X_test
    )

    detector = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    detector.fit(
        train_scaled
    )

    test_ood = -detector.decision_function(
        test_scaled
    )


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

    meta["OOD_Score"] = (
        test_ood
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

    meta["OODAware_Predicted"] = (
        ood_pred
    )

    meta["OODAware_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - ood_pred
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

ood_df = pd.DataFrame(
    ood_results
)

comparison_df = pd.concat(
    [
        baseline_df,
        ood_df
    ],
    ignore_index=True
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# MODEL SUMMARY
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
# OOD QUARTILES
# ============================================================

obs_df["OOD_Quartile"] = pd.qcut(
    obs_df["OOD_Score"],
    q=4,
    labels=[
        "Q1 - most typical",
        "Q2",
        "Q3",
        "Q4 - most unusual"
    ],
    duplicates="drop"
)


ood_summary = (
    obs_df
    .groupby(
        "OOD_Quartile",
        observed=True
    )
    .agg(

        Observations=(
            "Code",
            "count"
        ),

        Mean_OOD_Score=(
            "OOD_Score",
            "mean"
        ),

        Baseline_MAE=(
            "Baseline_Absolute_Error",
            "mean"
        ),

        OODAware_MAE=(
            "OODAware_Absolute_Error",
            "mean"
        ),

        Baseline_RMSE=(
            "Baseline_Absolute_Error",
            lambda x:
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        ),

        OODAware_RMSE=(
            "OODAware_Absolute_Error",
            lambda x:
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        )
    )
    .reset_index()
)


# ============================================================
# Q4-Q1 DEGRADATION
# ============================================================

q1 = ood_summary.iloc[0]
q4 = ood_summary.iloc[-1]


baseline_degradation = (
    (
        q4["Baseline_MAE"]
        - q1["Baseline_MAE"]
    )
    /
    q1["Baseline_MAE"]
    * 100
)


ood_degradation = (
    (
        q4["OODAware_MAE"]
        - q1["OODAware_MAE"]
    )
    /
    q1["OODAware_MAE"]
    * 100
)


# ============================================================
# PRINT
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
print("OOD QUARTILE PERFORMANCE")
print("================================================")

print(
    ood_summary.to_string(
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
    "OOD-aware:",
    round(
        ood_degradation,
        4
    ),
    "%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_df.to_csv(
    "YCEO_OODAware_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_OODAware_Model_Summary.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_OODAware_Observation_Results.csv",
    index=False
)

ood_summary.to_csv(
    "YCEO_OODAware_OOD_Quartiles.csv",
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
    "YCEO_OODAware_Model_Comparison.csv"
)

print(
    "YCEO_OODAware_Model_Summary.csv"
)

print(
    "YCEO_OODAware_Observation_Results.csv"
)

print(
    "YCEO_OODAware_OOD_Quartiles.csv"
)

print()
print(
    "OOD-aware XGBoost experiment complete."
)