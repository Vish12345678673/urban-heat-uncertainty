import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CODE_DIR = PROJECT_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import numpy as np
import pandas as pd

from utils.paths import (
    UNCERTAINTY_RESULTS_DIR
)

from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    UNCERTAINTY_RESULTS_DIR
    / "YCEO_UncertaintyVsOOD_Merged.csv"
)

RANDOM_STATE = 42

COVERAGE_LEVELS = [
    1.00,
    0.95,
    0.90,
    0.80,
    0.70,
    0.60,
    0.50,
    0.40,
    0.30
]

HIGH_ERROR_QUANTILE = 0.80


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
    c for c in required
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


print()
print("============================================================")
print("SELECTIVE PREDICTION BENCHMARK")
print("============================================================")

print(
    "Observations:",
    len(df)
)

print(
    "Folds:",
    df["Fold"].nunique()
)


# ============================================================
# HIGH-ERROR DEFINITION
# ============================================================

high_error_threshold = np.quantile(
    df["Absolute_Error"],
    HIGH_ERROR_QUANTILE
)


df["High_Error"] = (
    df["Absolute_Error"]
    >= high_error_threshold
).astype(int)


print()
print("============================================================")
print("HIGH-ERROR DEFINITION")
print("============================================================")

print(
    "Threshold:",
    round(
        high_error_threshold,
        6
    )
)

print(
    "High-error prevalence:",
    round(
        df["High_Error"].mean() * 100,
        3
    ),
    "%"
)


# ============================================================
# OOD SCORE DIRECTION
# ============================================================

y_event = df[
    "High_Error"
].to_numpy()


ood_values = df[
    "OOD_Score"
].to_numpy()


auc_positive = roc_auc_score(
    y_event,
    ood_values
)


auc_negative = roc_auc_score(
    y_event,
    -ood_values
)


if auc_positive >= auc_negative:

    ood_risk = ood_values

    ood_direction = (
        "higher_score_more_risk"
    )

else:

    ood_risk = -ood_values

    ood_direction = (
        "lower_score_more_risk"
    )


raw_risk = df[
    "Raw_Sigma"
].to_numpy()


calibrated_risk = df[
    "Calibrated_Sigma"
].to_numpy()


# ============================================================
# ALL-PREDICTION BASELINE
# ============================================================

baseline_mae = df[
    "Absolute_Error"
].mean()


baseline_rmse = np.sqrt(
    np.mean(
        (
            df["Observed"].to_numpy()
            -
            df["Predicted"].to_numpy()
        ) ** 2
    )
)


print()
print("============================================================")
print("FULL-COVERAGE BASELINE")
print("============================================================")

print(
    "MAE:",
    round(
        baseline_mae,
        6
    )
)

print(
    "RMSE:",
    round(
        baseline_rmse,
        6
    )
)


# ============================================================
# SELECTIVE RISK FUNCTION
# ============================================================
#
# Higher score = more risk.
#
# For a requested coverage c:
#
#   keep the lowest-risk c fraction
#
#   reject the highest-risk (1-c) fraction
#
# ============================================================

def evaluate_selective_risk(
    score,
    coverage
):

    n = len(score)


    n_keep = max(
        1,
        int(
            np.floor(
                n * coverage
            )
        )
    )


    order = np.argsort(
        score
    )


    retained_indices = order[
        :n_keep
    ]


    rejected_indices = order[
        n_keep:
    ]


    retained = df.iloc[
        retained_indices
    ]


    rejected = df.iloc[
        rejected_indices
    ]


    retained_mae = retained[
        "Absolute_Error"
    ].mean()


    retained_rmse = np.sqrt(
        np.mean(
            (
                retained[
                    "Observed"
                ].to_numpy()
                -
                retained[
                    "Predicted"
                ].to_numpy()
            ) ** 2
        )
    )


    rejected_mae = np.nan

    rejected_rmse = np.nan


    if len(rejected) > 0:

        rejected_mae = rejected[
            "Absolute_Error"
        ].mean()


        rejected_rmse = np.sqrt(
            np.mean(
                (
                    rejected[
                        "Observed"
                    ].to_numpy()
                    -
                    rejected[
                        "Predicted"
                    ].to_numpy()
                ) ** 2
            )
        )


    error_reduction = (
        (
            baseline_mae
            -
            retained_mae
        )
        /
        baseline_mae
        *
        100
    )


    return {
        "Coverage":
            coverage,

        "Retained_Observations":
            len(retained),

        "Rejected_Observations":
            len(rejected),

        "Selective_MAE":
            retained_mae,

        "Selective_RMSE":
            retained_rmse,

        "Rejected_MAE":
            rejected_mae,

        "Rejected_RMSE":
            rejected_rmse,

        "MAE_Reduction_Percent":
            error_reduction
    }


# ============================================================
# BUILD SELECTIVE RISK TABLE
# ============================================================

rows = []


for coverage in COVERAGE_LEVELS:

    ood_result = evaluate_selective_risk(
        ood_risk,
        coverage
    )


    raw_result = evaluate_selective_risk(
        raw_risk,
        coverage
    )


    calibrated_result = evaluate_selective_risk(
        calibrated_risk,
        coverage
    )


    rows.append({

        "Coverage":
            coverage,

        "Retained_Observations":
            ood_result[
                "Retained_Observations"
            ],

        "OOD_MAE":
            ood_result[
                "Selective_MAE"
            ],

        "Raw_Uncertainty_MAE":
            raw_result[
                "Selective_MAE"
            ],

        "Calibrated_Uncertainty_MAE":
            calibrated_result[
                "Selective_MAE"
            ],

        "OOD_RMSE":
            ood_result[
                "Selective_RMSE"
            ],

        "Raw_Uncertainty_RMSE":
            raw_result[
                "Selective_RMSE"
            ],

        "Calibrated_Uncertainty_RMSE":
            calibrated_result[
                "Selective_RMSE"
            ],

        "OOD_MAE_Reduction_Percent":
            ood_result[
                "MAE_Reduction_Percent"
            ],

        "Raw_Uncertainty_MAE_Reduction_Percent":
            raw_result[
                "MAE_Reduction_Percent"
            ],

        "Calibrated_Uncertainty_MAE_Reduction_Percent":
            calibrated_result[
                "MAE_Reduction_Percent"
            ]
    })


selective_df = pd.DataFrame(
    rows
)


print()
print("============================================================")
print("SELECTIVE RISK")
print("============================================================")

print(
    selective_df.to_string(
        index=False
    )
)


# ============================================================
# ERROR REJECTION
# ============================================================
#
# For each rejection fraction:
#
#   how much of the total high-error population is removed?
#
# ============================================================

capture_rows = []


for coverage in COVERAGE_LEVELS:

    rejection_fraction = (
        1.0 - coverage
    )


    n_reject = int(
        np.floor(
            len(df)
            * rejection_fraction
        )
    )


    if n_reject == 0:

        continue


    def evaluate_rejection(
        score
    ):

        order = np.argsort(
            score
        )[::-1
        ]

        rejected_indices = order[
            :n_reject
        ]


        rejected = df.iloc[
            rejected_indices
        ]


        high_error_removed = (
            rejected[
                "High_Error"
            ].sum()
        )


        total_high_error = (
            df["High_Error"].sum()
        )


        high_error_recall = (
            high_error_removed
            /
            total_high_error
        )


        return {
            "Rejected":
                len(rejected),

            "Rejected_MAE":
                rejected[
                    "Absolute_Error"
                ].mean(),

            "High_Error_Recall":
                high_error_recall
        }


    ood_result = evaluate_rejection(
        ood_risk
    )


    raw_result = evaluate_rejection(
        raw_risk
    )


    calibrated_result = evaluate_rejection(
        calibrated_risk
    )


    capture_rows.append({

        "Coverage":
            coverage,

        "Rejection_Percent":
            rejection_fraction * 100,

        "OOD_Rejected_MAE":
            ood_result[
                "Rejected_MAE"
            ],

        "Raw_Uncertainty_Rejected_MAE":
            raw_result[
                "Rejected_MAE"
            ],

        "Calibrated_Uncertainty_Rejected_MAE":
            calibrated_result[
                "Rejected_MAE"
            ],

        "OOD_High_Error_Recall":
            ood_result[
                "High_Error_Recall"
            ],

        "Raw_Uncertainty_High_Error_Recall":
            raw_result[
                "High_Error_Recall"
            ],

        "Calibrated_Uncertainty_High_Error_Recall":
            calibrated_result[
                "High_Error_Recall"
            ]
    })


capture_df = pd.DataFrame(
    capture_rows
)


print()
print("============================================================")
print("ERROR REJECTION")
print("============================================================")

print(
    capture_df.to_string(
        index=False
    )
)


# ============================================================
# EFFICIENCY OF REJECTION
# ============================================================
#
# Fraction of all absolute error removed per fraction of
# observations rejected.
# ============================================================

efficiency_rows = []


total_absolute_error = (
    df["Absolute_Error"]
    .sum()
)


for coverage in COVERAGE_LEVELS:

    rejection_fraction = (
        1.0 - coverage
    )


    n_reject = int(
        np.floor(
            len(df)
            * rejection_fraction
        )
    )


    if n_reject == 0:

        continue


    def error_capture(
        score
    ):

        order = np.argsort(
            score
        )[::-1
        ]

        rejected = df.iloc[
            order[
                :n_reject
            ]
        ]


        error_removed = (
            rejected[
                "Absolute_Error"
            ].sum()
        )


        share = (
            error_removed
            /
            total_absolute_error
        )


        return share


    efficiency_rows.append({

        "Coverage":
            coverage,

        "Rejection_Percent":
            rejection_fraction * 100,

        "OOD_Error_Share_Removed":
            error_capture(
                ood_risk
            ),

        "Raw_Uncertainty_Error_Share_Removed":
            error_capture(
                raw_risk
            ),

        "Calibrated_Uncertainty_Error_Share_Removed":
            error_capture(
                calibrated_risk
            )
    })


efficiency_df = pd.DataFrame(
    efficiency_rows
)


print()
print("============================================================")
print("ERROR CAPTURE EFFICIENCY")
print("============================================================")

print(
    efficiency_df.to_string(
        index=False
    )
)


# ============================================================
# WITHIN-FOLD SELECTIVE PERFORMANCE
# ============================================================
#
# Repeat the risk-coverage analysis independently in each
# spatial fold.
# ============================================================

fold_rows = []


for fold in sorted(
    df["Fold"].unique()
):

    fold_df = df[
        df["Fold"] == fold
    ].copy()


    fold_ood = (
        fold_df[
            "OOD_Score"
        ].to_numpy()
    )


    fold_y = (
        fold_df[
            "High_Error"
        ].to_numpy()
    )


    # Determine OOD direction inside the fold.

    pos = roc_auc_score(
        fold_y,
        fold_ood
    )

    neg = roc_auc_score(
        fold_y,
        -fold_ood
    )


    if pos >= neg:

        fold_ood_risk = fold_ood

    else:

        fold_ood_risk = -fold_ood


    fold_raw = (
        fold_df[
            "Raw_Sigma"
        ].to_numpy()
    )


    fold_cal = (
        fold_df[
            "Calibrated_Sigma"
        ].to_numpy()
    )


    coverage = 0.80


    n_keep = max(
        1,
        int(
            len(fold_df)
            * coverage
        )
    )


    def fold_selective_mae(
        score
    ):

        order = np.argsort(
            score
        )

        selected = order[
            :n_keep
        ]

        return fold_df.iloc[
            selected
        ][
            "Absolute_Error"
        ].mean()


    fold_rows.append({

        "Fold":
            fold,

        "Coverage":
            coverage,

        "OOD_MAE":
            fold_selective_mae(
                fold_ood_risk
            ),

        "Raw_Uncertainty_MAE":
            fold_selective_mae(
                fold_raw
            ),

        "Calibrated_Uncertainty_MAE":
            fold_selective_mae(
                fold_cal
            )
    })


fold_selective_df = pd.DataFrame(
    fold_rows
)


print()
print("============================================================")
print("WITHIN-FOLD SELECTIVE RISK AT 80% COVERAGE")
print("============================================================")

print(
    fold_selective_df.to_string(
        index=False
    )
)


# ============================================================
# MEAN WITHIN-FOLD REDUCTION
# ============================================================

fold_baseline_mae = []


for fold in sorted(
    df["Fold"].unique()
):

    temp = df[
        df["Fold"] == fold
    ]

    fold_baseline_mae.append(
        temp[
            "Absolute_Error"
        ].mean()
    )


fold_selective_df[
    "Baseline_MAE"
] = fold_basive_mae if False else np.nan


# Recompute safely.

baseline_map = {}


for fold in sorted(
    df["Fold"].unique()
):

    temp = df[
        df["Fold"] == fold
    ]

    baseline_map[
        fold
    ] = temp[
        "Absolute_Error"
    ].mean()


fold_selective_df[
    "Baseline_MAE"
] = fold_selective_df[
    "Fold"
].map(
    baseline_map
)


fold_selective_df[
    "OOD_Reduction_Percent"
] = (
    (
        fold_selective_df[
            "Baseline_MAE"
        ]
        -
        fold_selective_df[
            "OOD_MAE"
        ]
    )
    /
    fold_selective_df[
        "Baseline_MAE"
    ]
    * 100
)


fold_selective_df[
    "Raw_Uncertainty_Reduction_Percent"
] = (
    (
        fold_selective_df[
            "Baseline_MAE"
        ]
        -
        fold_selective_df[
            "Raw_Uncertainty_MAE"
        ]
    )
    /
    fold_selective_df[
        "Baseline_MAE"
    ]
    * 100
)


fold_selective_df[
    "Calibrated_Uncertainty_Reduction_Percent"
] = (
    (
        fold_selective_df[
            "Baseline_MAE"
        ]
        -
        fold_selective_df[
            "Calibrated_Uncertainty_MAE"
        ]
    )
    /
    fold_selective_df[
        "Baseline_MAE"
    ]
    * 100
)


print()
print("============================================================")
print("WITHIN-FOLD MAE REDUCTION")
print("============================================================")

print(
    fold_selective_df[
        [
            "Fold",
            "OOD_Reduction_Percent",
            "Raw_Uncertainty_Reduction_Percent",
            "Calibrated_Uncertainty_Reduction_Percent"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "Full_Coverage_MAE",
        "80pct_Coverage_OOD_MAE",
        "80pct_Coverage_Raw_Uncertainty_MAE",
        "80pct_Coverage_Calibrated_Uncertainty_MAE",
        "50pct_Coverage_OOD_MAE",
        "50pct_Coverage_Raw_Uncertainty_MAE",
        "50pct_Coverage_Calibrated_Uncertainty_MAE"
    ],

    "Value": [

        baseline_mae,

        selective_df.loc[
            selective_df["Coverage"] == 0.80,
            "OOD_MAE"
        ].iloc[0],

        selective_df.loc[
            selective_df["Coverage"] == 0.80,
            "Raw_Uncertainty_MAE"
        ].iloc[0],

        selective_df.loc[
            selective_df["Coverage"] == 0.80,
            "Calibrated_Uncertainty_MAE"
        ].iloc[0],

        selective_df.loc[
            selective_df["Coverage"] == 0.50,
            "OOD_MAE"
        ].iloc[0],

        selective_df.loc[
            selective_df["Coverage"] == 0.50,
            "Raw_Uncertainty_MAE"
        ].iloc[0],

        selective_df.loc[
            selective_df["Coverage"] == 0.50,
            "Calibrated_Uncertainty_MAE"
        ].iloc[0]
    ]
})


# ============================================================
# SAVE
# ============================================================

selective_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_SelectivePrediction_RiskCoverage.csv",
    index=False
)

capture_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_SelectivePrediction_ErrorRejection.csv",
    index=False
)

efficiency_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_SelectivePrediction_ErrorCaptureEfficiency.csv",
    index=False
)

fold_selective_df.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_SelectivePrediction_WithinFold.csv",
    index=False
)

summary.to_csv(UNCERTAINTY_RESULTS_DIR / "YCEO_SelectivePrediction_Summary.csv",
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
    "YCEO_SelectivePrediction_RiskCoverage.csv"
)

print(
    "YCEO_SelectivePrediction_ErrorRejection.csv"
)

print(
    "YCEO_SelectivePrediction_ErrorCaptureEfficiency.csv"
)

print(
    "YCEO_SelectivePrediction_WithinFold.csv"
)

print(
    "YCEO_SelectivePrediction_Summary.csv"
)

print()
print(
    "Selective prediction benchmark complete."
)


