import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from utils.paths import (
    UNCERTAINTY_RESULTS_DIR
)



# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    UNCERTAINTY_RESULTS_DIR
    / "YCEO_UncertaintyVsOOD_Merged.csv"
)

HIGH_ERROR_QUANTILE = 0.80

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


required = [
    "Fold",
    "Code",
    "Observed",
    "Predicted",
    "Absolute_Error",
    "OOD_Score",
    "Raw_Sigma",
    "Calibrated_Sigma"
]


missing = [
    c
    for c in required
    if c not in df.columns
]


if missing:

    raise ValueError(
        "Missing columns: "
        + str(missing)
    )


df = df.dropna(
    subset=[
        "Observed",
        "Predicted",
        "Absolute_Error",
        "OOD_Score",
        "Raw_Sigma",
        "Calibrated_Sigma"
    ]
).copy()


# ============================================================
# HIGH-ERROR EVENT
# ============================================================

error_threshold = np.quantile(
    df["Absolute_Error"],
    HIGH_ERROR_QUANTILE
)


df["High_Error"] = (
    df["Absolute_Error"]
    >= error_threshold
).astype(int)


print()
print("============================================================")
print("AURC SELECTIVE-RISK BENCHMARK")
print("============================================================")

print(
    "Observations:",
    len(df)
)

print(
    "Folds:",
    df["Fold"].nunique()
)

print(
    "High-error threshold:",
    round(
        error_threshold,
        6
    )
)


# ============================================================
# OOD DIRECTION
# ============================================================

def get_ood_risk(
    y_event,
    ood_values
):

    positive_auc = roc_auc_score(
        y_event,
        ood_values
    )

    negative_auc = roc_auc_score(
        y_event,
        -ood_values
    )

    if positive_auc >= negative_auc:

        return (
            ood_values,
            "higher_score_more_risk"
        )

    return (
        -ood_values,
        "lower_score_more_risk"
    )


# ============================================================
# AURC CALCULATION
# ============================================================
#
# For every possible retained observation:
#
#   rank by risk
#   retain the lowest-risk observations
#   calculate selective MAE
#
# We then integrate risk over coverage using the trapezoidal
# rule.
#
# At full coverage the risk equals ordinary MAE.
# ============================================================

def calculate_aurc(
    errors,
    risk
):

    n = len(errors)

    order = np.argsort(
        risk
    )

    sorted_errors = errors[
        order
    ]


    cumulative_error = np.cumsum(
        sorted_errors
    )


    cumulative_n = np.arange(
        1,
        n + 1
    )


    coverage = (
        cumulative_n
        /
        n
    )


    selective_risk = (
        cumulative_error
        /
        cumulative_n
    )


    # Add the zero-coverage origin.
    coverage_curve = np.concatenate(
        [
            np.array([0.0]),
            coverage
        ]
    )


    risk_curve = np.concatenate(
        [
            np.array([
                selective_risk[0]
            ]),
            selective_risk
        ]
    )


    aurc = np.trapezoid(
        risk_curve,
        coverage_curve
    )


    return (
        aurc,
        coverage_curve,
        risk_curve
    )


# ============================================================
# FOLD-LEVEL AURC
# ============================================================

fold_rows = []

curve_rows = []


for fold in sorted(
    df["Fold"].unique()
):

    fold_df = df[
        df["Fold"] == fold
    ].copy()


    errors = (
        fold_df[
            "Absolute_Error"
        ]
        .to_numpy()
    )


    y_event = (
        fold_df[
            "High_Error"
        ]
        .to_numpy()
    )


    ood_values = (
        fold_df[
            "OOD_Score"
        ]
        .to_numpy()
    )


    # --------------------------------------------------------
    # Determine OOD risk direction independently within fold.
    # --------------------------------------------------------

    ood_risk, ood_direction = (
        get_ood_risk(
            y_event,
            ood_values
        )
    )


    raw_risk = (
        fold_df[
            "Raw_Sigma"
        ]
        .to_numpy()
    )


    calibrated_risk = (
        fold_df[
            "Calibrated_Sigma"
        ]
        .to_numpy()
    )


    # --------------------------------------------------------
    # Calculate AURC.
    # --------------------------------------------------------

    ood_aurc, ood_cov, ood_risk_curve = (
        calculate_aurc(
            errors,
            ood_risk
        )
    )


    raw_aurc, raw_cov, raw_risk_curve = (
        calculate_aurc(
            errors,
            raw_risk
        )
    )


    cal_aurc, cal_cov, cal_risk_curve = (
        calculate_aurc(
            errors,
            calibrated_risk
        )
    )


    # --------------------------------------------------------
    # Oracle AURC.
    #
    # This ranks observations by their ACTUAL future error.
    # It is only a theoretical lower-bound reference.
    # It must never be presented as an achievable predictor.
    # --------------------------------------------------------

    oracle_risk = errors.copy()


    oracle_aurc, _, _ = calculate_aurc(
        errors,
        oracle_risk
    )


    # --------------------------------------------------------
    # Random-ranking reference.
    # Expected approximate AURC for random selection is related
    # to the full-sample mean error.
    # --------------------------------------------------------

    random_aurc = errors.mean()


    fold_rows.append({

        "Fold":
            fold,

        "Observations":
            len(fold_df),

        "Baseline_MAE":
            errors.mean(),

        "OOD_AURC":
            ood_aurc,

        "Raw_Uncertainty_AURC":
            raw_aurc,

        "Calibrated_Uncertainty_AURC":
            cal_aurc,

        "Oracle_AURC":
            oracle_aurc,

        "Random_AURC":
            random_aurc,

        "Raw_Improvement_vs_OOD":
            ood_aurc - raw_aurc,

        "Calibrated_Improvement_vs_OOD":
            ood_aurc - cal_aurc,

        "OOD_Direction":
            ood_direction
    })


    # --------------------------------------------------------
    # Store curves at selected coverage points.
    # --------------------------------------------------------

    selected_coverages = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00
    ]


    def interpolate_curve(
        coverage_curve,
        risk_curve,
        target
    ):

        return np.interp(
            target,
            coverage_curve,
            risk_curve
        )


    for c in selected_coverages:

        curve_rows.append({

            "Fold":
                fold,

            "Coverage":
                c,

            "OOD_MAE":
                interpolate_curve(
                    ood_cov,
                    ood_risk_curve,
                    c
                ),

            "Raw_Uncertainty_MAE":
                interpolate_curve(
                    raw_cov,
                    raw_risk_curve,
                    c
                ),

            "Calibrated_Uncertainty_MAE":
                interpolate_curve(
                    cal_cov,
                    cal_risk_curve,
                    c
                )
        })


fold_df = pd.DataFrame(
    fold_rows
)

curve_df = pd.DataFrame(
    curve_rows
)


# ============================================================
# PRINT FOLD RESULTS
# ============================================================

print()
print("============================================================")
print("FOLD-LEVEL AURC")
print("============================================================")

print(
    fold_df.to_string(
        index=False
    )
)


# ============================================================
# FIVE-FOLD SUMMARY
# ============================================================

summary = pd.DataFrame([

    {
        "Method":
            "OOD",

        "Mean_AURC":
            fold_df[
                "OOD_AURC"
            ].mean(),

        "SD_AURC":
            fold_df[
                "OOD_AURC"
            ].std(ddof=1)
    },

    {
        "Method":
            "Raw_Uncertainty",

        "Mean_AURC":
            fold_df[
                "Raw_Uncertainty_AURC"
            ].mean(),

        "SD_AURC":
            fold_df[
                "Raw_Uncertainty_AURC"
            ].std(ddof=1)
    },

    {
        "Method":
            "Calibrated_Uncertainty",

        "Mean_AURC":
            fold_df[
                "Calibrated_Uncertainty_AURC"
            ].mean(),

        "SD_AURC":
            fold_df[
                "Calibrated_Uncertainty_AURC"
            ].std(ddof=1)
    },

    {
        "Method":
            "Random",

        "Mean_AURC":
            fold_df[
                "Random_AURC"
            ].mean(),

        "SD_AURC":
            fold_df[
                "Random_AURC"
            ].std(ddof=1)
    },

    {
        "Method":
            "Oracle",

        "Mean_AURC":
            fold_df[
                "Oracle_AURC"
            ].mean(),

        "SD_AURC":
            fold_df[
                "Oracle_AURC"
            ].std(ddof=1)
    }
])


# ============================================================
# IMPROVEMENT
# ============================================================

ood_mean_aurc = summary.loc[
    summary["Method"] == "OOD",
    "Mean_AURC"
].iloc[0]


raw_mean_aurc = summary.loc[
    summary["Method"] == "Raw_Uncertainty",
    "Mean_AURC"
].iloc[0]


cal_mean_aurc = summary.loc[
    summary["Method"] == "Calibrated_Uncertainty",
    "Mean_AURC"
].iloc[0]


ood_to_raw_improvement = (
    ood_mean_aurc
    -
    raw_mean_aurc
)


ood_to_cal_improvement = (
    ood_mean_aurc
    -
    cal_mean_aurc
)


relative_raw_improvement = (
    ood_to_raw_improvement
    /
    ood_mean_aurc
    *
    100
)


relative_cal_improvement = (
    ood_to_cal_improvement
    /
    ood_mean_aurc
    *
    100
)


print()
print("============================================================")
print("AURC SUMMARY")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("UNCERTAINTY VS OOD AURC")
print("============================================================")

print(
    "OOD mean AURC:",
    round(
        ood_mean_aurc,
        6
    )
)

print(
    "Raw uncertainty mean AURC:",
    round(
        raw_mean_aurc,
        6
    )
)

print(
    "Calibrated uncertainty mean AURC:",
    round(
        cal_mean_aurc,
        6
    )
)

print()

print(
    "Raw AURC improvement:",
    round(
        ood_to_raw_improvement,
        6
    )
)

print(
    "Raw relative improvement:",
    round(
        relative_raw_improvement,
        3
    ),
    "%"
)

print()

print(
    "Calibrated AURC improvement:",
    round(
        ood_to_cal_improvement,
        6
    )
)

print(
    "Calibrated relative improvement:",
    round(
        relative_cal_improvement,
        3
    ),
    "%"
)


# ============================================================
# COVERAGE-LEVEL MEAN RISK
# ============================================================

coverage_summary = (
    curve_df
    .groupby(
        "Coverage"
    )
    .agg(

        OOD_MAE=(
            "OOD_MAE",
            "mean"
        ),

        Raw_Uncertainty_MAE=(
            "Raw_Uncertainty_MAE",
            "mean"
        ),

        Calibrated_Uncertainty_MAE=(
            "Calibrated_Uncertainty_MAE",
            "mean"
        )
    )
    .reset_index()
)


coverage_summary[
    "Raw_Improvement_vs_OOD"
] = (
    coverage_summary[
        "OOD_MAE"
    ]
    -
    coverage_summary[
        "Raw_Uncertainty_MAE"
    ]
)


coverage_summary[
    "Calibrated_Improvement_vs_OOD"
] = (
    coverage_summary[
        "OOD_MAE"
    ]
    -
    coverage_summary[
        "Calibrated_Uncertainty_MAE"
    ]
)


print()
print("============================================================")
print("MEAN SELECTIVE RISK BY COVERAGE")
print("============================================================")

print(
    coverage_summary.to_string(
        index=False
    )
)


# ============================================================
# WITHIN-FOLD WIN COUNT
# ============================================================

raw_wins = int(
    (
        fold_df[
            "Raw_Uncertainty_AURC"
        ]
        <
        fold_df[
            "OOD_AURC"
        ]
    ).sum()
)


cal_wins = int(
    (
        fold_df[
            "Calibrated_Uncertainty_AURC"
        ]
        <
        fold_df[
            "OOD_AURC"
        ]
    ).sum()
)


print()
print("============================================================")
print("AURC WIN COUNTS")
print("============================================================")

print(
    "Raw uncertainty beats OOD:",
    raw_wins,
    "/",
    len(fold_df)
)

print(
    "Calibrated uncertainty beats OOD:",
    cal_wins,
    "/",
    len(fold_df)
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_AURC_FoldResults.csv",
    index=False
)

summary.to_csv(
    "YCEO_AURC_Summary.csv",
    index=False
)

curve_df.to_csv(
    "YCEO_AURC_RiskCoverageCurve.csv",
    index=False
)

coverage_summary.to_csv(
    "YCEO_AURC_CoverageSummary.csv",
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
    "YCEO_AURC_FoldResults.csv"
)

print(
    "YCEO_AURC_Summary.csv"
)

print(
    "YCEO_AURC_RiskCoverageCurve.csv"
)

print(
    "YCEO_AURC_CoverageSummary.csv"
)

print()
print(
    "AURC benchmark complete."
)