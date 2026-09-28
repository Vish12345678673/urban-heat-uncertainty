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
    UNCERTAINTY_RESULTS_DIR,
    OOD_RESULTS_DIR
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

OOD_FILE = (
    OOD_RESULTS_DIR
    / "YCEO_OOD_ObservationResults.csv"
)

UNCERTAINTY_FILE = (
    UNCERTAINTY_RESULTS_DIR
    / "YCEO_NestedUncertainty_ObservationResults.csv"
)
RANDOM_STATE = 42

# Define a high-error observation as the top 20% of errors.
HIGH_ERROR_QUANTILE = 0.80

TOP_K_LEVELS = [
    0.10,
    0.20,
    0.30
]


# ============================================================
# LOAD
# ============================================================

print()
print("============================================================")
print("UNCERTAINTY VS OOD FAILURE DETECTION")
print("============================================================")

ood = pd.read_csv(
    OOD_FILE
)

uncertainty = pd.read_csv(
    UNCERTAINTY_FILE
)

print(
    "OOD observations:",
    len(ood)
)

print(
    "Uncertainty observations:",
    len(uncertainty)
)


# ============================================================
# FIND OOD SCORE COLUMN
# ============================================================

possible_ood_scores = [
    "OOD_Score",
    "OOD_score",
    "Mean_OOD_Score",
    "IsolationForest_Score",
    "OODScore"
]

ood_score_column = None

for column in possible_ood_scores:

    if column in ood.columns:

        ood_score_column = column

        break


if ood_score_column is None:

    candidates = [
        c
        for c in ood.columns
        if "ood" in c.lower()
        and pd.api.types.is_numeric_dtype(
            ood[c]
        )
    ]

    if len(candidates) == 0:

        raise ValueError(
            "Could not identify an OOD score column.\n"
            "Available columns:\n"
            + str(
                ood.columns.tolist()
            )
        )

    ood_score_column = candidates[0]


print(
    "OOD score column:",
    ood_score_column
)


# ============================================================
# CHECK UNCERTAINTY COLUMNS
# ============================================================

required_uncertainty = [
    "Fold",
    "Code",
    "Observed",
    "Predicted",
    "Absolute_Error",
    "Raw_Sigma",
    "Calibrated_Sigma"
]

missing_uncertainty = [
    c
    for c in required_uncertainty
    if c not in uncertainty.columns
]

if missing_uncertainty:

    raise ValueError(
        "Missing uncertainty columns: "
        + str(
            missing_uncertainty
        )
    )


# ============================================================
# CHECK OOD IDENTIFIERS
# ============================================================

required_ood = [
    "Fold",
    "Code"
]

missing_ood = [
    c
    for c in required_ood
    if c not in ood.columns
]

if missing_ood:

    raise ValueError(
        "Missing OOD identifier columns: "
        + str(
            missing_ood
        )
    )


# ============================================================
# BUILD CLEAN OOD TABLE
# ============================================================

ood_clean = (
    ood[
        [
            "Fold",
            "Code",
            ood_score_column
        ]
    ]
    .copy()
)


ood_clean = (
    ood_clean
    .drop_duplicates(
        subset=[
            "Fold",
            "Code"
        ]
    )
)


ood_clean = ood_clean.rename(
    columns={
        ood_score_column:
            "OOD_Score"
    }
)


# ============================================================
# BUILD CLEAN UNCERTAINTY TABLE
# ============================================================

unc_clean = uncertainty[
    [
        "Fold",
        "Code",
        "Observed",
        "Predicted",
        "Absolute_Error",
        "Raw_Sigma",
        "Calibrated_Sigma"
    ]
].copy()


unc_clean = (
    unc_clean
    .drop_duplicates(
        subset=[
            "Fold",
            "Code"
        ]
    )
)


# ============================================================
# MERGE
# ============================================================

merged = pd.merge(

    unc_clean,

    ood_clean,

    on=[
        "Fold",
        "Code"
    ],

    how="inner"
)


print()
print(
    "Matched observations:",
    len(merged)
)


if len(merged) == 0:

    raise RuntimeError(
        "No observations matched between OOD and "
        "nested uncertainty files."
    )


# ============================================================
# MISSING VALUE CHECK
# ============================================================

required_numeric = [
    "Absolute_Error",
    "Raw_Sigma",
    "Calibrated_Sigma",
    "OOD_Score"
]


missing_counts = (
    merged[
        required_numeric
    ]
    .isna()
    .sum()
)


if missing_counts.sum() > 0:

    print()
    print(
        "Missing values before filtering:"
    )

    print(
        missing_counts
        .to_string()
    )


    merged = merged.dropna(
        subset=required_numeric
    ).copy()


print(
    "Usable observations:",
    len(merged)
)


# ============================================================
# HIGH-ERROR LABEL
# ============================================================
#
# Top 20% of actual prediction errors.
# This label is ONLY used to evaluate detection.
# No model is fitted with it.
# ============================================================

error_threshold = np.quantile(
    merged[
        "Absolute_Error"
    ],
    HIGH_ERROR_QUANTILE
)


merged[
    "High_Error"
] = (

    merged[
        "Absolute_Error"
    ]
    >=
    error_threshold

).astype(int)


print()
print("============================================================")
print("HIGH-ERROR DEFINITION")
print("============================================================")

print(
    "Absolute-error threshold:",
    round(
        error_threshold,
        6
    )
)

print(
    "High-error observations:",
    int(
        merged["High_Error"].sum()
    )
)

print(
    "High-error prevalence:",
    round(
        merged["High_Error"].mean() * 100,
        3
    ),
    "%"
)


# ============================================================
# OOD SCORE DIRECTION
# ============================================================
#
# Some anomaly scores use:
#
#   larger = more anomalous
#
# while others use:
#
#   smaller = more anomalous
#
# Test both directions.
# ============================================================

y_event = (
    merged[
        "High_Error"
    ]
    .to_numpy()
)


ood_values = (
    merged[
        "OOD_Score"
    ]
    .to_numpy()
)


ood_auc_positive = roc_auc_score(
    y_event,
    ood_values
)


ood_auc_negative = roc_auc_score(
    y_event,
    -ood_values
)


if (
    ood_auc_positive
    >=
    ood_auc_negative
):

    ood_detection_score = (
        ood_values
    )

    ood_direction = (
        "higher_score_more_risk"
    )

    ood_auc = (
        ood_auc_positive
    )

else:

    ood_detection_score = (
        -ood_values
    )

    ood_direction = (
        "lower_score_more_risk"
    )

    ood_auc = (
        ood_auc_negative
    )


print()
print(
    "OOD score direction:",
    ood_direction
)


# ============================================================
# UNCERTAINTY SCORES
# ============================================================

raw_uncertainty = (
    merged[
        "Raw_Sigma"
    ]
    .to_numpy()
)


calibrated_uncertainty = (
    merged[
        "Calibrated_Sigma"
    ]
    .to_numpy()
)


# ============================================================
# AUROC
# ============================================================

raw_auc = roc_auc_score(
    y_event,
    raw_uncertainty
)


calibrated_auc = roc_auc_score(
    y_event,
    calibrated_uncertainty
)


# ============================================================
# AUPRC
# ============================================================

ood_auprc = average_precision_score(
    y_event,
    ood_detection_score
)


raw_auprc = average_precision_score(
    y_event,
    raw_uncertainty
)


calibrated_auprc = average_precision_score(
    y_event,
    calibrated_uncertainty
)


# ============================================================
# PRINT DETECTION PERFORMANCE
# ============================================================

print()
print("============================================================")
print("FAILURE DETECTION PERFORMANCE")
print("============================================================")

print(
    "OOD AUROC:",
    round(
        ood_auc,
        6
    )
)

print(
    "Raw uncertainty AUROC:",
    round(
        raw_auc,
        6
    )
)

print(
    "Calibrated uncertainty AUROC:",
    round(
        calibrated_auc,
        6
    )
)


print()
print(
    "OOD AUPRC:",
    round(
        ood_auprc,
        6
    )
)

print(
    "Raw uncertainty AUPRC:",
    round(
        raw_auprc,
        6
    )
)

print(
    "Calibrated uncertainty AUPRC:",
    round(
        calibrated_auprc,
        6
    )
)


# ============================================================
# SPEARMAN CORRELATION WITH ERROR
# ============================================================

ood_spearman, ood_spearman_p = (
    spearmanr(
        ood_detection_score,
        merged[
            "Absolute_Error"
        ]
    )
)


raw_spearman, raw_spearman_p = (
    spearmanr(
        raw_uncertainty,
        merged[
            "Absolute_Error"
        ]
    )
)


cal_spearman, cal_spearman_p = (
    spearmanr(
        calibrated_uncertainty,
        merged[
            "Absolute_Error"
        ]
    )
)


print()
print("============================================================")
print("SPEARMAN ASSOCIATION WITH ABSOLUTE ERROR")
print("============================================================")

print(
    "OOD:",
    round(
        ood_spearman,
        6
    ),
    "p =",
    ood_spearman_p
)

print(
    "Raw uncertainty:",
    round(
        raw_spearman,
        6
    ),
    "p =",
    raw_spearman_p
)

print(
    "Calibrated uncertainty:",
    round(
        cal_spearman,
        6
    ),
    "p =",
    cal_spearman_p
)


# ============================================================
# TOP-K PRECISION
# ============================================================
#
# Select the highest-risk observations according to each
# detector and measure how many are genuinely high-error.
# ============================================================

precision_rows = []


for fraction in TOP_K_LEVELS:

    n = max(
        1,
        int(
            len(merged)
            * fraction
        )
    )


    def top_precision(
        score
    ):

        order = np.argsort(
            score
        )[::-1]


        selected = order[
            :n
        ]


        return (
            y_event[
                selected
            ]
            .mean()
        )


    precision_rows.append({

        "Top_Fraction":
            fraction,

        "Observations_Selected":
            n,

        "Base_High_Error_Prevalence":
            y_event.mean(),

        "OOD_Precision":
            top_precision(
                ood_detection_score
            ),

        "Raw_Uncertainty_Precision":
            top_precision(
                raw_uncertainty
            ),

        "Calibrated_Uncertainty_Precision":
            top_precision(
                calibrated_uncertainty
            )
    })


precision_df = pd.DataFrame(
    precision_rows
)


print()
print("============================================================")
print("TOP-K HIGH-ERROR PRECISION")
print("============================================================")

print(
    precision_df.to_string(
        index=False
    )
)


# ============================================================
# HIGH-ERROR RECALL / CAPTURE
# ============================================================

high_error_indices = np.where(
    y_event == 1
)[0]


capture_rows = []


for fraction in TOP_K_LEVELS:

    n = max(
        1,
        int(
            len(merged)
            * fraction
        )
    )


    def capture_rate(
        score
    ):

        order = np.argsort(
            score
        )[::-1]


        selected = set(
            order[
                :n
            ]
        )


        captured = sum(
            index in selected
            for index in high_error_indices
        )


        return (
            captured
            /
            len(
                high_error_indices
            )
        )


    capture_rows.append({

        "Top_Fraction":
            fraction,

        "OOD_Recall":
            capture_rate(
                ood_detection_score
            ),

        "Raw_Uncertainty_Recall":
            capture_rate(
                raw_uncertainty
            ),

        "Calibrated_Uncertainty_Recall":
            capture_rate(
                calibrated_uncertainty
            )
    })


capture_df = pd.DataFrame(
    capture_rows
)


print()
print("============================================================")
print("HIGH-ERROR CAPTURE")
print("============================================================")

print(
    capture_df.to_string(
        index=False
    )
)


# ============================================================
# WITHIN-FOLD AUROC
# ============================================================

fold_rows = []


for fold in sorted(
    merged[
        "Fold"
    ].unique()
):

    temp = merged[
        merged["Fold"]
        ==
        fold
    ]


    y_fold = (
        temp[
            "High_Error"
        ]
        .to_numpy()
    )


    if len(
        np.unique(
            y_fold
        )
    ) < 2:

        continue


    fold_ood_values = (
        temp[
            "OOD_Score"
        ]
        .to_numpy()
    )


    fold_auc_positive = (
        roc_auc_score(
            y_fold,
            fold_ood_values
        )
    )


    fold_auc_negative = (
        roc_auc_score(
            y_fold,
            -fold_ood_values
        )
    )


    fold_ood_auc = max(
        fold_auc_positive,
        fold_auc_negative
    )


    fold_raw_auc = (
        roc_auc_score(
            y_fold,
            temp[
                "Raw_Sigma"
            ]
        )
    )


    fold_calibrated_auc = (
        roc_auc_score(
            y_fold,
            temp[
                "Calibrated_Sigma"
            ]
        )
    )


    fold_rows.append({

        "Fold":
            fold,

        "Observations":
            len(temp),

        "OOD_AUROC":
            fold_ood_auc,

        "Raw_Uncertainty_AUROC":
            fold_raw_auc,

        "Calibrated_Uncertainty_AUROC":
            fold_calibrated_auc
    })


fold_df = pd.DataFrame(
    fold_rows
)


print()
print("============================================================")
print("WITHIN-FOLD AUROC")
print("============================================================")

print(
    fold_df.to_string(
        index=False
    )
)


# ============================================================
# WITHIN-FOLD MEAN
# ============================================================

if len(fold_df) > 0:

    print()
    print(
        "Mean within-fold OOD AUROC:",
        round(
            fold_df[
                "OOD_AUROC"
            ].mean(),
            6
        )
    )

    print(
        "Mean within-fold raw "
        "uncertainty AUROC:",
        round(
            fold_df[
                "Raw_Uncertainty_AUROC"
            ].mean(),
            6
        )
    )

    print(
        "Mean within-fold calibrated "
        "uncertainty AUROC:",
        round(
            fold_df[
                "Calibrated_Uncertainty_AUROC"
            ].mean(),
            6
        )
    )


# ============================================================
# IMPROVEMENT OVER OOD
# ============================================================

print()
print("============================================================")
print("UNCERTAINTY IMPROVEMENT OVER OOD")
print("============================================================")

print(
    "Raw AUROC improvement:",
    round(
        raw_auc - ood_auc,
        6
    )
)

print(
    "Calibrated AUROC improvement:",
    round(
        calibrated_auc - ood_auc,
        6
    )
)

print(
    "Raw AUPRC improvement:",
    round(
        raw_auprc - ood_auprc,
        6
    )
)

print(
    "Calibrated AUPRC improvement:",
    round(
        calibrated_auprc - ood_auprc,
        6
    )
)


# ============================================================
# DETECTION COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame([

    {
        "Method":
            "OOD",

        "AUROC":
            ood_auc,

        "AUPRC":
            ood_auprc,

        "Spearman":
            ood_spearman,

        "Spearman_p":
            ood_spearman_p
    },

    {
        "Method":
            "Raw_Uncertainty",

        "AUROC":
            raw_auc,

        "AUPRC":
            raw_auprc,

        "Spearman":
            raw_spearman,

        "Spearman_p":
            raw_spearman_p
    },

    {
        "Method":
            "Calibrated_Uncertainty",

        "AUROC":
            calibrated_auc,

        "AUPRC":
            calibrated_auprc,

        "Spearman":
            cal_spearman,

        "Spearman_p":
            cal_spearman_p
    }
])


# ============================================================
# SAVE
# ============================================================

merged.to_csv(
    "YCEO_UncertaintyVsOOD_Merged.csv",
    index=False
)

precision_df.to_csv(
    "YCEO_UncertaintyVsOOD_TopKPrecision.csv",
    index=False
)

capture_df.to_csv(
    "YCEO_UncertaintyVsOOD_HighErrorCapture.csv",
    index=False
)

fold_df.to_csv(
    "YCEO_UncertaintyVsOOD_WithinFoldAUROC.csv",
    index=False
)

comparison_df.to_csv(
    "YCEO_UncertaintyVsOOD_Comparison.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_UncertaintyVsOOD_Merged.csv"
)

print(
    "YCEO_UncertaintyVsOOD_TopKPrecision.csv"
)

print(
    "YCEO_UncertaintyVsOOD_HighErrorCapture.csv"
)

print(
    "YCEO_UncertaintyVsOOD_WithinFoldAUROC.csv"
)

print(
    "YCEO_UncertaintyVsOOD_Comparison.csv"
)

print()
print(
    "Uncertainty vs OOD detection analysis complete."
)