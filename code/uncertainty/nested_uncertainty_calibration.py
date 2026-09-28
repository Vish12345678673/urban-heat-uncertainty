import os
import random

import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT / "code")
)

from utils.paths import (
    DATA_DIR,
    UNCERTAINTY_RESULTS_DIR
)

DATA_FILE = (
    DATA_DIR
    / "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"
)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from scipy.stats import spearmanr

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    UNCERTAINTY_RESULTS_DIR
    / "YCEO_Uncertainty_ObservationResults.csv"
)

RANDOM_STATE = 42

N_OUTER_FOLDS = 5

N_CALIBRATION_FOLDS = 4

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

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


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
    (df["NDVI"] >= -1)
    &
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
print("STRICT NESTED UNCERTAINTY CALIBRATION")
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

print(
    "Outer folds:",
    N_OUTER_FOLDS
)

print(
    "Internal calibration folds:",
    N_CALIBRATION_FOLDS
)


# ============================================================
# MODEL
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
# BUILD OUT-OF-FOLD RESIDUALS INSIDE OUTER TRAINING SET
# ============================================================
#
# This generates residual targets for uncertainty calibration.
#
# Every residual is produced by a model that did NOT train on
# the corresponding observation.
#
# The inner folds are spatial GroupKFold folds.
# ============================================================

def generate_internal_oof_residuals(
    X_outer_train,
    y_outer_train,
    groups_outer_train
):

    inner_cv = GroupKFold(
        n_splits=N_CALIBRATION_FOLDS
    )

    oof_prediction = np.full(
        len(X_outer_train),
        np.nan,
        dtype=float
    )


    for inner_fold, (
        inner_train_idx,
        inner_valid_idx
    ) in enumerate(

        inner_cv.split(
            X_outer_train,
            y_outer_train,
            groups=groups_outer_train
        ),

        start=1
    ):

        model = make_model(
            seed=RANDOM_STATE + inner_fold
        )


        model.fit(
            X_outer_train.iloc[
                inner_train_idx
            ],

            y_outer_train.iloc[
                inner_train_idx
            ]
        )


        oof_prediction[
            inner_valid_idx
        ] = model.predict(

            X_outer_train.iloc[
                inner_valid_idx
            ]
        )


    if not np.isfinite(
        oof_prediction
    ).all():

        raise RuntimeError(
            "Internal OOF predictions contain invalid values."
        )


    residual = (
        y_outer_train.to_numpy()
        -
        oof_prediction
    )


    squared_residual = (
        residual ** 2
    )


    absolute_residual = np.abs(
        residual
    )


    return (
        oof_prediction,
        residual,
        squared_residual,
        absolute_residual
    )


# ============================================================
# FIT FINAL MEAN MODEL
# ============================================================

def fit_mean_model(
    X_train,
    y_train
):

    model = make_model(
        seed=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# FIT UNCERTAINTY MODEL
# ============================================================
#
# The uncertainty target comes from INTERNAL OOF residuals.
#
# Therefore the uncertainty model is not trained on in-sample
# residuals.
# ============================================================

def fit_uncertainty_model(
    X_outer_train,
    y_outer_train,
    groups_outer_train
):

    (
        oof_prediction,
        residual,
        squared_residual,
        absolute_residual
    ) = generate_internal_oof_residuals(

        X_outer_train,
        y_outer_train,
        groups_outer_train
    )


    # --------------------------------------------------------
    # Clip only extreme residual targets.
    # Threshold is calculated inside outer training only.
    # --------------------------------------------------------

    clip_value = np.quantile(
        squared_residual,
        0.99
    )


    variance_target = np.minimum(
        squared_residual,
        clip_value
    )


    # Log transform.
    log_variance_target = np.log1p(
        variance_target
    )


    variance_model = make_model(
        seed=RANDOM_STATE + 1000
    )


    variance_model.fit(
        X_outer_train,
        log_variance_target
    )


    return (
        variance_model,
        clip_value
    )


# ============================================================
# CALIBRATE SIGMA USING INNER OOF DATA
# ============================================================
#
# We want a calibration map from raw predicted sigma to
# empirical absolute error.
#
# The calibrator is fitted only on OOF predictions/residuals
# generated from the outer training set.
#
# We use quantile binning rather than an external library so
# the procedure is completely transparent.
# ============================================================

def fit_sigma_calibrator(
    X_outer_train,
    y_outer_train,
    groups_outer_train,
    variance_model
):

    (
        oof_prediction,
        residual,
        squared_residual,
        absolute_residual
    ) = generate_internal_oof_residuals(

        X_outer_train,
        y_outer_train,
        groups_outer_train
    )


    raw_log_variance = (
        variance_model.predict(
            X_outer_train
        )
    )


    raw_variance = np.expm1(
        raw_log_variance
    )


    raw_variance = np.maximum(
        raw_variance,
        1e-8
    )


    raw_sigma = np.sqrt(
        raw_variance
    )


    calibration_frame = pd.DataFrame({

        "Raw_Sigma":
            raw_sigma,

        "Absolute_Error":
            absolute_residual

    })


    calibration_frame[
        "Sigma_Bin"
    ] = pd.qcut(

        calibration_frame[
            "Raw_Sigma"
        ],

        q=20,

        labels=False,

        duplicates="drop"
    )


    bin_table = (

        calibration_frame

        .groupby(
            "Sigma_Bin",
            observed=True
        )

        .agg(

            Mean_Raw_Sigma=(
                "Raw_Sigma",
                "mean"
            ),

            Mean_Absolute_Error=(
                "Absolute_Error",
                "mean"
            ),

            Observations=(
                "Raw_Sigma",
                "count"
            )
        )

        .reset_index()
    )


    # Ensure monotonic calibration mapping.
    #
    # Isotonic regression is preferable here because we do not
    # assume a particular parametric relationship.

    from sklearn.isotonic import IsotonicRegression


    isotonic = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip"
    )


    isotonic.fit(

        bin_table[
            "Mean_Raw_Sigma"
        ],

        bin_table[
            "Mean_Absolute_Error"
        ],

        sample_weight=bin_table[
            "Observations"
        ]
    )


    return (
        isotonic,
        bin_table
    )


# ============================================================
# CREATE CALIBRATED TEST SIGMA
# ============================================================

def calibrated_test_sigma(
    X_test,
    variance_model,
    isotonic
):

    raw_log_variance = (
        variance_model.predict(
            X_test
        )
    )


    raw_variance = np.expm1(
        raw_log_variance
    )


    raw_variance = np.maximum(
        raw_variance,
        1e-8
    )


    raw_sigma = np.sqrt(
        raw_variance
    )


    calibrated_abs_error = (
        isotonic.predict(
            raw_sigma
        )
    )


    # --------------------------------------------------------
    # Convert expected absolute error into Gaussian-like sigma:
    #
    # E|e| ≈ sigma * sqrt(2/pi)
    #
    # This is a scale conversion only.
    # --------------------------------------------------------

    calibrated_sigma = (
        calibrated_abs_error
        /
        np.sqrt(
            2.0 / np.pi
        )
    )


    calibrated_sigma = np.maximum(
        calibrated_sigma,
        1e-6
    )


    return (
        raw_sigma,
        calibrated_sigma
    )


# ============================================================
# INTERVAL METRICS
# ============================================================

def interval_metrics(
    observed,
    prediction,
    sigma,
    z
):

    lower = (
        prediction
        -
        z * sigma
    )

    upper = (
        prediction
        +
        z * sigma
    )


    covered = (
        (observed >= lower)
        &
        (observed <= upper)
    )


    coverage = covered.mean()

    width = np.mean(
        upper - lower
    )


    return (
        coverage,
        width
    )


# ============================================================
# OUTER CV
# ============================================================

outer_cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)


fold_results = []

all_observations = []

all_calibration_bins = []


for fold, (
    outer_train_idx,
    outer_test_idx
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


    X_outer_train = X.iloc[
        outer_train_idx
    ].copy()

    y_outer_train = y.iloc[
        outer_train_idx
    ].copy()

    groups_outer_train = groups.iloc[
        outer_train_idx
    ].copy()


    X_test = X.iloc[
        outer_test_idx
    ].copy()

    y_test = y.iloc[
        outer_test_idx
    ].copy()


    # ========================================================
    # MEAN MODEL
    # ========================================================

    mean_model = fit_mean_model(
        X_outer_train,
        y_outer_train
    )


    test_prediction = mean_model.predict(
        X_test
    )


    # ========================================================
    # UNCERTAINTY MODEL
    # ========================================================

    (
        variance_model,
        variance_clip
    ) = fit_uncertainty_model(

        X_outer_train,
        y_outer_train,
        groups_outer_train
    )


    # ========================================================
    # CALIBRATOR
    # ========================================================

    (
        isotonic,
        calibration_bins
    ) = fit_sigma_calibrator(

        X_outer_train,
        y_outer_train,
        groups_outer_train,
        variance_model
    )


    calibration_bins[
        "OuterFold"
    ] = fold


    all_calibration_bins.append(
        calibration_bins
    )


    # ========================================================
    # TEST UNCERTAINTY
    # ========================================================

    (
        raw_sigma,
        calibrated_sigma
    ) = calibrated_test_sigma(

        X_test,
        variance_model,
        isotonic
    )


    observed = y_test.to_numpy()


    absolute_error = np.abs(
        observed
        -
        test_prediction
    )


    squared_error = (
        observed
        -
        test_prediction
    ) ** 2


    # ========================================================
    # POINT PREDICTION
    # ========================================================

    r2 = r2_score(
        y_test,
        test_prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_prediction
        )
    )

    mae = mean_absolute_error(
        y_test,
        test_prediction
    )


    # ========================================================
    # COVERAGE
    # ========================================================

    coverage_80, width_80 = interval_metrics(
        observed,
        test_prediction,
        calibrated_sigma,
        1.281551565545
    )


    coverage_90, width_90 = interval_metrics(
        observed,
        test_prediction,
        calibrated_sigma,
        1.644853626951
    )


    coverage_95, width_95 = interval_metrics(
        observed,
        test_prediction,
        calibrated_sigma,
        1.959963984540
    )


    print()
    print("POINT PREDICTION")

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
    print("CALIBRATED INTERVAL COVERAGE")

    print(
        "80% =",
        round(
            coverage_80 * 100,
            3
        ),
        "%",
        "width =",
        round(
            width_80,
            6
        )
    )

    print(
        "90% =",
        round(
            coverage_90 * 100,
            3
        ),
        "%",
        "width =",
        round(
            width_90,
            6
        )
    )

    print(
        "95% =",
        round(
            coverage_95 * 100,
            3
        ),
        "%",
        "width =",
        round(
            width_95,
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

        "Coverage_80":
            coverage_80,

        "Coverage_90":
            coverage_90,

        "Coverage_95":
            coverage_95,

        "Width_80":
            width_80,

        "Width_90":
            width_90,

        "Width_95":
            width_95,

        "Mean_Raw_Sigma":
            raw_sigma.mean(),

        "Mean_Calibrated_Sigma":
            calibrated_sigma.mean()
    })


    # ========================================================
    # OBSERVATIONS
    # ========================================================

    meta = df.iloc[
        outer_test_idx
    ][
        [
            "Code",
            "Lat",
            "Lon",
            "spatial_block"
        ]
    ].copy()


    meta["Fold"] = fold

    meta["Observed"] = observed

    meta["Predicted"] = (
        test_prediction
    )

    meta["Absolute_Error"] = (
        absolute_error
    )

    meta["Squared_Error"] = (
        squared_error
    )

    meta["Raw_Sigma"] = (
        raw_sigma
    )

    meta["Calibrated_Sigma"] = (
        calibrated_sigma
    )


    for level, z in [
        ("80", 1.281551565545),
        ("90", 1.644853626951),
        ("95", 1.959963984540)
    ]:

        lower = (
            test_prediction
            - z * calibrated_sigma
        )

        upper = (
            test_prediction
            + z * calibrated_sigma
        )


        meta[
            f"Lower_{level}"
        ] = lower

        meta[
            f"Upper_{level}"
        ] = upper

        meta[
            f"Covered_{level}"
        ] = (
            (observed >= lower)
            &
            (observed <= upper)
        )


    all_observations.append(
        meta
    )


# ============================================================
# FINAL DATAFRAMES
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

obs_df = pd.concat(
    all_observations,
    ignore_index=True
)

calibration_bins_df = pd.concat(
    all_calibration_bins,
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "R2",
        "RMSE",
        "MAE",
        "Coverage_80",
        "Coverage_90",
        "Coverage_95"
    ],

    "Mean": [

        fold_df["R2"].mean(),

        fold_df["RMSE"].mean(),

        fold_df["MAE"].mean(),

        fold_df["Coverage_80"].mean(),

        fold_df["Coverage_90"].mean(),

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

        fold_df["Coverage_80"].std(
            ddof=1
        ),

        fold_df["Coverage_90"].std(
            ddof=1
        ),

        fold_df["Coverage_95"].std(
            ddof=1
        )
    ]
})


# ============================================================
# UNCERTAINTY / ERROR ASSOCIATION
# ============================================================

raw_pearson = np.corrcoef(
    obs_df["Raw_Sigma"],
    obs_df["Absolute_Error"]
)[0, 1]


cal_pearson = np.corrcoef(
    obs_df["Calibrated_Sigma"],
    obs_df["Absolute_Error"]
)[0, 1]


raw_spearman, raw_spearman_p = spearmanr(
    obs_df["Raw_Sigma"],
    obs_df["Absolute_Error"]
)


cal_spearman, cal_spearman_p = spearmanr(
    obs_df["Calibrated_Sigma"],
    obs_df["Absolute_Error"]
)


print()
print("============================================================")
print("UNCERTAINTY / ERROR ASSOCIATION")
print("============================================================")

print(
    "Raw Pearson:",
    round(
        raw_pearson,
        6
    )
)

print(
    "Calibrated Pearson:",
    round(
        cal_pearson,
        6
    )
)

print(
    "Raw Spearman:",
    round(
        raw_spearman,
        6
    ),
    "p =",
    raw_spearman_p
)

print(
    "Calibrated Spearman:",
    round(
        cal_spearman,
        6
    ),
    "p =",
    cal_spearman_p
)


# ============================================================
# CALIBRATED UNCERTAINTY QUARTILES
# ============================================================

obs_df[
    "Calibrated_Uncertainty_Quartile"
] = pd.qcut(

    obs_df[
        "Calibrated_Sigma"
    ],

    q=4,

    labels=[
        "Q1 - lowest",
        "Q2",
        "Q3",
        "Q4 - highest"
    ],

    duplicates="drop"
)


quartile_df = (

    obs_df

    .groupby(
        "Calibrated_Uncertainty_Quartile",
        observed=True
    )

    .agg(

        Observations=(
            "Code",
            "count"
        ),

        Mean_Calibrated_Sigma=(
            "Calibrated_Sigma",
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
        )
    )

    .reset_index()
)


print()
print("============================================================")
print("ERROR BY CALIBRATED UNCERTAINTY QUARTILE")
print("============================================================")

print(
    quartile_df.to_string(
        index=False
    )
)


# ============================================================
# Q4/Q1 DEGRADATION
# ============================================================

q1 = quartile_df.iloc[0]

q4 = quartile_df.iloc[-1]


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
print("Q4 VS Q1 CALIBRATED RISK")
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
# RISK-COVERAGE
# ============================================================

sorted_df = obs_df.sort_values(
    "Calibrated_Sigma"
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
        len(sorted_df)
        * coverage_level
    )


    retained = sorted_df.iloc[
        :n_keep
    ]


    risk_rows.append({

        "Coverage":
            coverage_level,

        "Observations_Retained":
            len(retained),

        "Mean_Calibrated_Sigma":
            retained[
                "Calibrated_Sigma"
            ].mean(),

        "Mean_Absolute_Error":
            retained[
                "Absolute_Error"
            ].mean(),

        "RMSE":
            np.sqrt(
                retained[
                    "Squared_Error"
                ].mean()
            )
    })


risk_df = pd.DataFrame(
    risk_rows
)


print()
print("============================================================")
print("CALIBRATED RISK-COVERAGE")
print("============================================================")

print(
    risk_df.to_string(
        index=False
    )
)


# ============================================================
# TOP 20% ERROR CONCENTRATION
# ============================================================

n_high = int(
    len(sorted_df) * 0.20
)


high_risk = sorted_df.iloc[
    -n_high:
]


error_share = (
    high_risk[
        "Absolute_Error"
    ].sum()
    /
    obs_df[
        "Absolute_Error"
    ].sum()
)


print()
print("============================================================")
print("HIGHEST-RISK ERROR CONCENTRATION")
print("============================================================")

print(
    "Highest 20% risk observations:",
    n_high
)

print(
    "Share of total absolute error:",
    round(
        error_share * 100,
        3
    ),
    "%"
)


# ============================================================
# COVERAGE CALIBRATION ERROR
# ============================================================

nominal = np.array([
    0.80,
    0.90,
    0.95
])


observed = np.array([
    fold_df["Coverage_80"].mean(),
    fold_df["Coverage_90"].mean(),
    fold_df["Coverage_95"].mean()
])


coverage_mae = np.mean(
    np.abs(
        observed
        -
        nominal
    )
)


coverage_max_error = np.max(
    np.abs(
        observed
        -
        nominal
    )
)


print()
print("============================================================")
print("CALIBRATION ERROR")
print("============================================================")

print(
    "Mean absolute coverage error:",
    round(
        coverage_mae,
        6
    )
)

print(
    "Maximum coverage error:",
    round(
        coverage_max_error,
        6
    )
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_NestedUncertainty_FoldResults.csv",
    index=False
)

obs_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_NestedUncertainty_ObservationResults.csv",
    index=False
)

calibration_bins_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_NestedUncertainty_CalibrationBins.csv",
    index=False
)

summary.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_NestedUncertainty_Summary.csv",
    index=False
)

quartile_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_NestedUncertainty_Quartiles.csv",
    index=False
)

risk_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_NestedUncertainty_RiskCoverage.csv",
    index=False
)


calibration_metrics = pd.DataFrame([{

    "Raw_Pearson":
        raw_pearson,

    "Calibrated_Pearson":
        cal_pearson,

    "Raw_Spearman":
        raw_spearman,

    "Raw_Spearman_p":
        raw_spearman_p,

    "Calibrated_Spearman":
        cal_spearman,

    "Calibrated_Spearman_p":
        cal_spearman_p,

    "Q4_Q1_MAE_Degradation":
        q4_q1_degradation,

    "High_Risk_20_Error_Share":
        error_share,

    "Mean_Coverage_Error":
        coverage_mae,

    "Maximum_Coverage_Error":
        coverage_max_error
}])


calibration_metrics.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_NestedUncertainty_CalibrationMetrics.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_NestedUncertainty_FoldResults.csv"
)

print(
    "YCEO_NestedUncertainty_ObservationResults.csv"
)

print(
    "YCEO_NestedUncertainty_CalibrationBins.csv"
)

print(
    "YCEO_NestedUncertainty_Summary.csv"
)

print(
    "YCEO_NestedUncertainty_Quartiles.csv"
)

print(
    "YCEO_NestedUncertainty_RiskCoverage.csv"
)

print(
    "YCEO_NestedUncertainty_CalibrationMetrics.csv"
)

print()
print(
    "Strict nested uncertainty calibration complete."
)

