import os
import random

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "code")
)

from utils.paths import (
    DATA_DIR,
    UNCERTAINTY_RESULTS_DIR
)
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = (
    DATA_DIR
    / "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"
)

RANDOM_STATE = 42

N_OUTER_FOLDS = 5

N_INTERNAL_FOLDS = 4

INTERNAL_VAL_SIZE = 0.15

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
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(
    RANDOM_STATE
)

random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)


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


X = df[FEATURES].copy()

y = df[TARGET].copy()

groups = df["spatial_block"].copy()


print()
print("============================================================")
print("UNCERTAINTY-AWARE XGBOOST")
print("============================================================")

print(
    "Dataset:",
    df.shape
)

print(
    "Features:",
    len(FEATURES)
)

print(
    "Spatial blocks:",
    groups.nunique()
)


# ============================================================
# MODEL FACTORY
# ============================================================

def make_model(
    seed=RANDOM_STATE
):

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

        random_state=seed,

        n_jobs=-1,

        tree_method="hist"
    )


# ============================================================
# LEAKAGE-FREE OOF RESIDUALS
# ============================================================
#
# The variance model must NOT be trained on residuals obtained
# from predictions made by a model trained on the same samples.
#
# Therefore, within each OUTER TRAINING SET:
#
#   inner spatial folds
#       ↓
#   OOF mean prediction
#       ↓
#   squared residual
#
# These OOF squared residuals become the target for the
# uncertainty model.
# ============================================================

def build_oof_residual_targets(
    X_train,
    y_train,
    groups_train
):

    inner_cv = GroupKFold(
        n_splits=N_INTERNAL_FOLDS
    )


    oof_prediction = np.full(
        len(X_train),
        np.nan,
        dtype=float
    )


    for inner_fold, (
        inner_train_idx,
        inner_valid_idx
    ) in enumerate(

        inner_cv.split(
            X_train,
            y_train,
            groups=groups_train
        ),

        start=1

    ):

        model = make_model(
            seed=RANDOM_STATE + inner_fold
        )


        X_inner_train = (
            X_train.iloc[
                inner_train_idx
            ]
        )

        y_inner_train = (
            y_train.iloc[
                inner_train_idx
            ]
        )

        X_inner_valid = (
            X_train.iloc[
                inner_valid_idx
            ]
        )


        model.fit(
            X_inner_train,
            y_inner_train
        )


        oof_prediction[
            inner_valid_idx
        ] = model.predict(
            X_inner_valid
        )


    valid_mask = np.isfinite(
        oof_prediction
    )


    if not valid_mask.all():

        raise RuntimeError(
            "OOF residual generation failed."
        )


    residual = (
        y_train.to_numpy()
        -
        oof_prediction
    )


    squared_residual = (
        residual ** 2
    )


    return (
        oof_prediction,
        residual,
        squared_residual
    )


# ============================================================
# FIT FINAL MEAN + UNCERTAINTY MODELS
# ============================================================

def fit_uncertainty_models(
    X_train,
    y_train,
    groups_train
):

    # --------------------------------------------------------
    # OOF residual target
    # --------------------------------------------------------

    (
        oof_prediction,
        oof_residual,
        squared_residual
    ) = build_oof_residual_targets(
        X_train,
        y_train,
        groups_train
    )


    # --------------------------------------------------------
    # Mean model on ALL outer-training observations
    # --------------------------------------------------------

    mean_model = make_model(
        seed=RANDOM_STATE
    )

    mean_model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Clip squared residuals to prevent a few massive errors
    # from completely dominating variance fitting.
    #
    # This clipping threshold is calculated using training data
    # only.
    # --------------------------------------------------------

    upper = np.quantile(
        squared_residual,
        0.99
    )


    variance_target = np.minimum(
        squared_residual,
        upper
    )


    # Log variance target improves numerical stability.
    log_variance_target = np.log1p(
        variance_target
    )


    # --------------------------------------------------------
    # Variance model
    # --------------------------------------------------------

    variance_model = make_model(
        seed=RANDOM_STATE + 100
    )

    variance_model.fit(
        X_train,
        log_variance_target
    )


    return (
        mean_model,
        variance_model,
        upper
    )


# ============================================================
# OUTER SPATIAL CV
# ============================================================

outer_cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)


fold_results = []

observation_results = []


for fold, (
    train_idx,
    test_idx
) in enumerate(

    outer_cv.split(
        X,
        y,
        groups=groups
    ),

    start=1
):

    print()
    print("============================================================")
    print(
        "OUTER SPATIAL FOLD",
        fold
    )
    print("============================================================")


    X_train = X.iloc[
        train_idx
    ].copy()

    y_train = y.iloc[
        train_idx
    ].copy()

    groups_train = groups.iloc[
        train_idx
    ].copy()


    X_test = X.iloc[
        test_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    # ========================================================
    # FIT MEAN + VARIANCE MODELS
    # ========================================================

    (
        mean_model,
        variance_model,
        variance_clip
    ) = fit_uncertainty_models(

        X_train,
        y_train,
        groups_train
    )


    # ========================================================
    # TEST PREDICTIONS
    # ========================================================

    mean_prediction = mean_model.predict(
        X_test
    )


    predicted_log_variance = (
        variance_model.predict(
            X_test
        )
    )


    predicted_variance = (
        np.expm1(
            predicted_log_variance
        )
    )


    predicted_variance = np.maximum(
        predicted_variance,
        1e-8
    )


    predicted_sigma = np.sqrt(
        predicted_variance
    )


    # ========================================================
    # CLIP EXTREME UNCERTAINTY FOR STABILITY
    # ========================================================

    max_sigma = np.sqrt(
        max(
            variance_clip,
            1e-8
        )
    )


    predicted_sigma = np.minimum(
        predicted_sigma,
        5.0 * max_sigma
    )


    # ========================================================
    # METRICS
    # ========================================================

    prediction_error = (
        y_test.to_numpy()
        -
        mean_prediction
    )


    absolute_error = np.abs(
        prediction_error
    )


    squared_error = (
        prediction_error ** 2
    )


    r2 = r2_score(
        y_test,
        mean_prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            mean_prediction
        )
    )


    mae = mean_absolute_error(
        y_test,
        mean_prediction
    )


    # ========================================================
    # COVERAGE
    # ========================================================
    #
    # Approximate 95% prediction interval:
    #
    #   prediction ± 1.96 sigma
    #
    # This is a diagnostic, not a claim of calibrated Gaussian
    # uncertainty.
    # ========================================================

    lower_95 = (
        mean_prediction
        -
        1.96 * predicted_sigma
    )


    upper_95 = (
        mean_prediction
        +
        1.96 * predicted_sigma
    )


    coverage_95 = np.mean(
        (
            y_test.to_numpy()
            >= lower_95
        )
        &
        (
            y_test.to_numpy()
            <= upper_95
        )
    )


    interval_width = np.mean(
        upper_95
        -
        lower_95
    )


    # ========================================================
    # PRINT PREDICTION RESULTS
    # ========================================================

    print()
    print("PREDICTION PERFORMANCE")

    print(
        "R²   =",
        round(
            r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            mae,
            6
        )
    )


    print()
    print("UNCERTAINTY")

    print(
        "95% coverage =",
        round(
            coverage_95 * 100,
            3
        ),
        "%"
    )

    print(
        "Mean interval width =",
        round(
            interval_width,
            6
        )
    )


    # ========================================================
    # STORE FOLD
    # ========================================================

    fold_results.append({

        "Fold":
            fold,

        "R2":
            r2,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "Coverage_95":
            coverage_95,

        "Mean_Interval_Width":
            interval_width,

        "Mean_Predicted_Sigma":
            predicted_sigma.mean()
    })


    # ========================================================
    # STORE OBSERVATIONS
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


    meta["Predicted"] = (
        mean_prediction
    )


    meta["Absolute_Error"] = (
        absolute_error
    )


    meta["Squared_Error"] = (
        squared_error
    )


    meta["Predicted_Sigma"] = (
        predicted_sigma
    )


    meta["Predicted_Variance"] = (
        predicted_variance
    )


    meta["Lower_95"] = (
        lower_95
    )


    meta["Upper_95"] = (
        upper_95
    )


    meta["Covered_95"] = (
        (
            y_test.to_numpy()
            >= lower_95
        )
        &
        (
            y_test.to_numpy()
            <= upper_95
        )
    )


    observation_results.append(
        meta
    )


# ============================================================
# COMBINE
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# UNCERTAINTY / ERROR ASSOCIATION
# ============================================================

pearson_r = np.corrcoef(
    obs_df["Predicted_Sigma"],
    obs_df["Absolute_Error"]
)[0, 1]


spearman_r = (
    obs_df[
        [
            "Predicted_Sigma",
            "Absolute_Error"
        ]
    ]
    .rank()
    .corr()
    .iloc[0, 1]
)


print()
print("============================================================")
print("UNCERTAINTY / ERROR ASSOCIATION")
print("============================================================")

print(
    "Pearson r :",
    round(
        pearson_r,
        6
    )
)

print(
    "Spearman r:",
    round(
        spearman_r,
        6
    )
)


# ============================================================
# UNCERTAINTY QUARTILES
# ============================================================

obs_df["Uncertainty_Quartile"] = pd.qcut(

    obs_df["Predicted_Sigma"],

    q=4,

    labels=[
        "Q1 - lowest uncertainty",
        "Q2",
        "Q3",
        "Q4 - highest uncertainty"
    ],

    duplicates="drop"
)


uncertainty_summary = (
    obs_df
    .groupby(
        "Uncertainty_Quartile",
        observed=True
    )
    .agg(

        Observations=(
            "Code",
            "count"
        ),

        Mean_Predicted_Sigma=(
            "Predicted_Sigma",
            "mean"
        ),

        Mean_Absolute_Error=(
            "Absolute_Error",
            "mean"
        ),

        RMSE=(
            "Squared_Error",
            lambda x:
            np.sqrt(
                np.mean(x)
            )
        ),

        Coverage_95=(
            "Covered_95",
            "mean"
        )
    )
    .reset_index()
)


print()
print("============================================================")
print("ERROR BY PREDICTED UNCERTAINTY QUARTILE")
print("============================================================")

print(
    uncertainty_summary.to_string(
        index=False
    )
)


# ============================================================
# Q4 VS Q1 ERROR DEGRADATION
# ============================================================

q1 = uncertainty_summary.iloc[0]

q4 = uncertainty_summary.iloc[-1]


q4_q1_degradation = (

    (
        q4["Mean_Absolute_Error"]
        -
        q1["Mean_Absolute_Error"]
    )
    /
    q1["Mean_Absolute_Error"]
    *
    100
)


print()
print("============================================================")
print("Q4 VS Q1 UNCERTAINTY DEGRADATION")
print("============================================================")

print(
    "MAE degradation:",
    round(
        q4_q1_degradation,
        4
    ),
    "%"
)


# ============================================================
# RISK-COVERAGE ANALYSIS
# ============================================================
#
# Sort observations from LOW to HIGH predicted uncertainty.
#
# At each retained fraction, calculate the average actual error.
#
# If uncertainty is useful, removing the highest-uncertainty
# observations should preferentially remove high-error cases.
# ============================================================

sorted_obs = obs_df.sort_values(
    "Predicted_Sigma"
).reset_index(
    drop=True
)


coverage_levels = [
    1.00,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50
]


risk_rows = []


for coverage_level in coverage_levels:

    n_keep = int(
        len(sorted_obs)
        * coverage_level
    )


    retained = sorted_obs.iloc[
        :n_keep
    ]


    risk_rows.append({

        "Coverage":
            coverage_level,

        "Observations_Retained":
            len(retained),

        "Mean_Predicted_Sigma":
            retained["Predicted_Sigma"].mean(),

        "Mean_Absolute_Error":
            retained["Absolute_Error"].mean(),

        "RMSE":
            np.sqrt(
                retained["Squared_Error"].mean()
            )
    })


risk_df = pd.DataFrame(
    risk_rows
)


print()
print("============================================================")
print("RISK-COVERAGE")
print("============================================================")

print(
    risk_df.to_string(
        index=False
    )
)


# ============================================================
# HIGH-RISK CONCENTRATION
# ============================================================
#
# What fraction of all absolute error lies in the highest 20%
# uncertainty observations?
# ============================================================

n_high = int(
    len(sorted_obs)
    * 0.20
)


high_uncertainty = sorted_obs.iloc[
    -n_high:
]


total_error = (
    obs_df["Absolute_Error"].sum()
)


high_error = (
    high_uncertainty[
        "Absolute_Error"
    ].sum()
)


high_error_share = (
    high_error
    /
    total_error
)


print()
print("============================================================")
print("HIGH-UNCERTAINTY ERROR CONCENTRATION")
print("============================================================")

print(
    "Highest 20% uncertainty observations:",
    len(high_uncertainty)
)

print(
    "Share of total absolute error:",
    round(
        high_error_share * 100,
        3
    ),
    "%"
)


# ============================================================
# FOLD SUMMARY
# ============================================================

cv_summary = pd.DataFrame({

    "Metric": [
        "R2",
        "RMSE",
        "MAE",
        "Coverage_95"
    ],

    "Mean": [

        fold_df["R2"].mean(),

        fold_df["RMSE"].mean(),

        fold_df["MAE"].mean(),

        fold_df["Coverage_95"].mean()
    ],

    "SD": [

        fold_df["R2"].std(
            ddof=1
        ),

        fold_df["RMSE"].std(
            ddof=1
        ),

        fold_df["MAE"].std(
            ddof=1
        ),

        fold_df["Coverage_95"].std(
            ddof=1
        )
    ]
})


print()
print("============================================================")
print("FIVE-FOLD SUMMARY")
print("============================================================")

print(
    cv_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_Uncertainty_FoldResults.csv",
    index=False
)

obs_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_Uncertainty_ObservationResults.csv",
    index=False
)

uncertainty_summary.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_Uncertainty_ErrorByQuartile.csv",
    index=False
)

risk_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_Uncertainty_RiskCoverage.csv",
    index=False
)

cv_summary.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_Uncertainty_Summary.csv",
    index=False
)


# ============================================================
# TEXT SUMMARY
# ============================================================

summary_text = f"""
UNCERTAINTY-AWARE XGBOOST SUMMARY

Five-fold spatial CV:
R2   = {fold_df["R2"].mean():.6f} ± {fold_df["R2"].std(ddof=1):.6f}
RMSE = {fold_df["RMSE"].mean():.6f} ± {fold_df["RMSE"].std(ddof=1):.6f}
MAE  = {fold_df["MAE"].mean():.6f} ± {fold_df["MAE"].std(ddof=1):.6f}

95% interval coverage:
{fold_df["Coverage_95"].mean() * 100:.3f}%

Uncertainty-error Pearson correlation:
{pearson_r:.6f}

Uncertainty-error Spearman correlation:
{spearman_r:.6f}

Q4/Q1 uncertainty-based MAE degradation:
{q4_q1_degradation:.4f}%

Highest-uncertainty 20% share of total absolute error:
{high_error_share * 100:.3f}%
"""


with open(
    "YCEO_Uncertainty_Report.txt",
    "w"
) as f:

    f.write(
        summary_text
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_Uncertainty_FoldResults.csv"
)

print(
    "YCEO_Uncertainty_ObservationResults.csv"
)

print(
    "YCEO_Uncertainty_ErrorByQuartile.csv"
)

print(
    "YCEO_Uncertainty_RiskCoverage.csv"
)

print(
    "YCEO_Uncertainty_Summary.csv"
)

print(
    "YCEO_Uncertainty_Report.txt"
)

print()
print(
    "Uncertainty-aware XGBoost experiment complete."
)
