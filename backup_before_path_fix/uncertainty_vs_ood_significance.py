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
    UNCERTAINTY_RESULTS_DIR
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    UNCERTAINTY_RESULTS_DIR
    / "YCEO_UncertaintyVsOOD_Merged.csv"
)

RANDOM_STATE = 42

N_BOOTSTRAPS = 5000
N_PERMUTATIONS = 5000

HIGH_ERROR_QUANTILE = 0.80

TOP_K_LEVELS = [
    0.10,
    0.20,
    0.30
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

required = [
    "Fold",
    "Code",
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
        "Absolute_Error",
        "OOD_Score",
        "Raw_Sigma",
        "Calibrated_Sigma"
    ]
).copy()


print()
print("============================================================")
print("PAIRED SIGNIFICANCE TEST: UNCERTAINTY VS OOD")
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

error_threshold = np.quantile(
    df["Absolute_Error"],
    HIGH_ERROR_QUANTILE
)

df["High_Error"] = (
    df["Absolute_Error"]
    >= error_threshold
).astype(int)

y = df["High_Error"].to_numpy()


print()
print("============================================================")
print("HIGH-ERROR DEFINITION")
print("============================================================")

print(
    "Threshold:",
    round(
        error_threshold,
        6
    )
)

print(
    "High-error observations:",
    int(y.sum())
)

print(
    "Prevalence:",
    round(
        y.mean() * 100,
        3
    ),
    "%"
)


# ============================================================
# SCORE ARRAYS
# ============================================================

ood_values = df[
    "OOD_Score"
].to_numpy()

raw_uncertainty = df[
    "Raw_Sigma"
].to_numpy()

calibrated_uncertainty = df[
    "Calibrated_Sigma"
].to_numpy()


# ============================================================
# OOD SCORE DIRECTION
# ============================================================

auc_positive = roc_auc_score(
    y,
    ood_values
)

auc_negative = roc_auc_score(
    y,
    -ood_values
)

if auc_positive >= auc_negative:

    ood_score = ood_values

    ood_direction = (
        "higher_score_more_risk"
    )

else:

    ood_score = -ood_values

    ood_direction = (
        "lower_score_more_risk"
    )


# ============================================================
# BASELINE PERFORMANCE
# ============================================================

ood_auc = roc_auc_score(
    y,
    ood_score
)

raw_auc = roc_auc_score(
    y,
    raw_uncertainty
)

calibrated_auc = roc_auc_score(
    y,
    calibrated_uncertainty
)


ood_auprc = average_precision_score(
    y,
    ood_score
)

raw_auprc = average_precision_score(
    y,
    raw_uncertainty
)

calibrated_auprc = average_precision_score(
    y,
    calibrated_uncertainty
)


observed_raw_auc_diff = (
    raw_auc - ood_auc
)

observed_cal_auc_diff = (
    calibrated_auc - ood_auc
)

observed_raw_auprc_diff = (
    raw_auprc - ood_auprc
)

observed_cal_auprc_diff = (
    calibrated_auprc - ood_auprc
)


print()
print("============================================================")
print("OBSERVED DETECTION PERFORMANCE")
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

print()

print(
    "OOD score direction:",
    ood_direction
)


# ============================================================
# SPEARMAN
# ============================================================

ood_spearman, ood_spearman_p = spearmanr(
    ood_score,
    df["Absolute_Error"]
)

raw_spearman, raw_spearman_p = spearmanr(
    raw_uncertainty,
    df["Absolute_Error"]
)

cal_spearman, cal_spearman_p = spearmanr(
    calibrated_uncertainty,
    df["Absolute_Error"]
)


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================
#
# Every bootstrap sample resamples observations jointly.
#
# Therefore OOD, raw uncertainty and calibrated uncertainty
# always remain paired with the same high-error labels.
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)

bootstrap_raw_auc_diff = []
bootstrap_cal_auc_diff = []

bootstrap_raw_auprc_diff = []
bootstrap_cal_auprc_diff = []


n = len(df)


for _ in range(
    N_BOOTSTRAPS
):

    indices = rng.integers(
        0,
        n,
        size=n
    )

    y_b = y[indices]

    if len(
        np.unique(y_b)
    ) < 2:
        continue

    ood_b = ood_score[
        indices
    ]

    raw_b = raw_uncertainty[
        indices
    ]

    cal_b = calibrated_uncertainty[
        indices
    ]


    auc_ood_b = roc_auc_score(
        y_b,
        ood_b
    )

    auc_raw_b = roc_auc_score(
        y_b,
        raw_b
    )

    auc_cal_b = roc_auc_score(
        y_b,
        cal_b
    )


    ap_ood_b = average_precision_score(
        y_b,
        ood_b
    )

    ap_raw_b = average_precision_score(
        y_b,
        raw_b
    )

    ap_cal_b = average_precision_score(
        y_b,
        cal_b
    )


    bootstrap_raw_auc_diff.append(
        auc_raw_b - auc_ood_b
    )

    bootstrap_cal_auc_diff.append(
        auc_cal_b - auc_ood_b
    )

    bootstrap_raw_auprc_diff.append(
        ap_raw_b - ap_ood_b
    )

    bootstrap_cal_auprc_diff.append(
        ap_cal_b - ap_ood_b
    )


bootstrap_raw_auc_diff = np.asarray(
    bootstrap_raw_auc_diff
)

bootstrap_cal_auc_diff = np.asarray(
    bootstrap_cal_auc_diff
)

bootstrap_raw_auprc_diff = np.asarray(
    bootstrap_raw_auprc_diff
)

bootstrap_cal_auprc_diff = np.asarray(
    bootstrap_cal_auprc_diff
)


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_ci(
    values
):

    lower = np.percentile(
        values,
        2.5
    )

    upper = np.percentile(
        values,
        97.5
    )

    return lower, upper


raw_auc_ci = bootstrap_ci(
    bootstrap_raw_auc_diff
)

cal_auc_ci = bootstrap_ci(
    bootstrap_cal_auc_diff
)

raw_auprc_ci = bootstrap_ci(
    bootstrap_raw_auprc_diff
)

cal_auprc_ci = bootstrap_ci(
    bootstrap_cal_auprc_diff
)


# ============================================================
# PAIRED PERMUTATION TEST
# ============================================================
#
# For every observation, randomly swap which score is considered
# detector A vs detector B.
#
# This preserves the paired structure.
# ============================================================

permutation_raw_auc = []
permutation_cal_auc = []

permutation_raw_auprc = []
permutation_cal_auprc = []


for _ in range(
    N_PERMUTATIONS
):

    swap = (
        rng.random(n)
        < 0.5
    )


    # --------------------------------------------------------
    # Raw uncertainty vs OOD
    # --------------------------------------------------------

    perm_raw = np.where(
        swap,
        ood_score,
        raw_uncertainty
    )

    perm_ood = np.where(
        swap,
        raw_uncertainty,
        ood_score
    )


    perm_auc_raw = (
        roc_auc_score(
            y,
            perm_raw
        )
    )

    perm_auc_ood = (
        roc_auc_score(
            y,
            perm_ood
        )
    )


    perm_ap_raw = (
        average_precision_score(
            y,
            perm_raw
        )
    )

    perm_ap_ood = (
        average_precision_score(
            y,
            perm_ood
        )
    )


    permutation_raw_auc.append(
        perm_auc_raw - perm_auc_ood
    )

    permutation_raw_auprc.append(
        perm_ap_raw - perm_ap_ood
    )


    # --------------------------------------------------------
    # Calibrated uncertainty vs OOD
    # --------------------------------------------------------

    perm_cal = np.where(
        swap,
        ood_score,
        calibrated_uncertainty
    )

    perm_ood_cal = np.where(
        swap,
        calibrated_uncertainty,
        ood_score
    )


    perm_auc_cal = (
        roc_auc_score(
            y,
            perm_cal
        )
    )

    perm_auc_ood_cal = (
        roc_auc_score(
            y,
            perm_ood_cal
        )
    )


    perm_ap_cal = (
        average_precision_score(
            y,
            perm_cal
        )
    )

    perm_ap_ood_cal = (
        average_precision_score(
            y,
            perm_ood_cal
        )
    )


    permutation_cal_auc.append(
        perm_auc_cal
        - perm_auc_ood_cal
    )

    permutation_cal_auprc.append(
        perm_ap_cal
        - perm_ap_ood_cal
    )


permutation_raw_auc = np.asarray(
    permutation_raw_auc
)

permutation_cal_auc = np.asarray(
    permutation_cal_auc
)

permutation_raw_auprc = np.asarray(
    permutation_raw_auprc
)

permutation_cal_auprc = np.asarray(
    permutation_cal_auprc
)


# ============================================================
# PERMUTATION P-VALUES
# ============================================================

raw_auc_p = np.mean(
    np.abs(
        permutation_raw_auc
    )
    >= abs(
        observed_raw_auc_diff
    )
)

cal_auc_p = np.mean(
    np.abs(
        permutation_cal_auc
    )
    >= abs(
        observed_cal_auc_diff
    )
)

raw_auprc_p = np.mean(
    np.abs(
        permutation_raw_auprc
    )
    >= abs(
        observed_raw_auprc_diff
    )
)

cal_auprc_p = np.mean(
    np.abs(
        permutation_cal_auprc
    )
    >= abs(
        observed_cal_auprc_diff
    )
)


# ============================================================
# PRINT SIGNIFICANCE
# ============================================================

print()
print("============================================================")
print("PAIRED BOOTSTRAP + PERMUTATION RESULTS")
print("============================================================")


print()
print("RAW UNCERTAINTY vs OOD")

print(
    "ΔAUROC:",
    round(
        observed_raw_auc_diff,
        6
    )
)

print(
    "95% bootstrap CI:",
    (
        round(
            raw_auc_ci[0],
            6
        ),
        round(
            raw_auc_ci[1],
            6
        )
    )
)

print(
    "Permutation p-value:",
    raw_auc_p
)


print()

print(
    "ΔAUPRC:",
    round(
        observed_raw_auprc_diff,
        6
    )
)

print(
    "95% bootstrap CI:",
    (
        round(
            raw_auprc_ci[0],
            6
        ),
        round(
            raw_auprc_ci[1],
            6
        )
    )
)

print(
    "Permutation p-value:",
    raw_auprc_p
)


print()
print(
    "CALIBRATED UNCERTAINTY vs OOD"
)

print(
    "ΔAUROC:",
    round(
        observed_cal_auc_diff,
        6
    )
)

print(
    "95% bootstrap CI:",
    (
        round(
            cal_auc_ci[0],
            6
        ),
        round(
            cal_auc_ci[1],
            6
        )
    )
)

print(
    "Permutation p-value:",
    cal_auc_p
)


print()

print(
    "ΔAUPRC:",
    round(
        observed_cal_auprc_diff,
        6
    )
)

print(
    "95% bootstrap CI:",
    (
        round(
            cal_auprc_ci[0],
            6
        ),
        round(
            cal_auprc_ci[1],
            6
        )
    )
)

print(
    "Permutation p-value:",
    cal_auprc_p
)


# ============================================================
# SPEARMAN RESULTS
# ============================================================

print()
print("============================================================")
print("ERROR ASSOCIATION")
print("============================================================")

print(
    "OOD Spearman:",
    round(
        ood_spearman,
        6
    ),
    "p =",
    ood_spearman_p
)

print(
    "Raw uncertainty Spearman:",
    round(
        raw_spearman,
        6
    ),
    "p =",
    raw_spearman_p
)

print(
    "Calibrated uncertainty Spearman:",
    round(
        cal_spearman,
        6
    ),
    "p =",
    cal_spearman_p
)


# ============================================================
# WITHIN-FOLD COMPARISON
# ============================================================

fold_rows = []


for fold in sorted(
    df["Fold"].unique()
):

    fold_df = df[
        df["Fold"] == fold
    ]


    y_fold = (
        fold_df[
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
        fold_df[
            "OOD_Score"
        ]
        .to_numpy()
    )


    positive_auc = roc_auc_score(
        y_fold,
        fold_ood_values
    )

    negative_auc = roc_auc_score(
        y_fold,
        -fold_ood_values
    )


    if positive_auc >= negative_auc:

        fold_ood_score = (
            fold_ood_values
        )

    else:

        fold_ood_score = (
            -fold_ood_values
        )


    fold_raw = (
        fold_df[
            "Raw_Sigma"
        ]
        .to_numpy()
    )

    fold_cal = (
        fold_df[
            "Calibrated_Sigma"
        ]
        .to_numpy()
    )


    auc_ood_fold = roc_auc_score(
        y_fold,
        fold_ood_score
    )

    auc_raw_fold = roc_auc_score(
        y_fold,
        fold_raw
    )

    auc_cal_fold = roc_auc_score(
        y_fold,
        fold_cal
    )


    ap_ood_fold = average_precision_score(
        y_fold,
        fold_ood_score
    )

    ap_raw_fold = average_precision_score(
        y_fold,
        fold_raw
    )

    ap_cal_fold = average_precision_score(
        y_fold,
        fold_cal
    )


    fold_rows.append({

        "Fold": fold,

        "Observations": len(
            fold_df
        ),

        "OOD_AUROC":
            auc_ood_fold,

        "Raw_Uncertainty_AUROC":
            auc_raw_fold,

        "Calibrated_Uncertainty_AUROC":
            auc_cal_fold,

        "Raw_Delta_AUROC":
            auc_raw_fold
            - auc_ood_fold,

        "Calibrated_Delta_AUROC":
            auc_cal_fold
            - auc_ood_fold,

        "OOD_AUPRC":
            ap_ood_fold,

        "Raw_Uncertainty_AUPRC":
            ap_raw_fold,

        "Calibrated_Uncertainty_AUPRC":
            ap_cal_fold,

        "Raw_Delta_AUPRC":
            ap_raw_fold
            - ap_ood_fold,

        "Calibrated_Delta_AUPRC":
            ap_cal_fold
            - ap_ood_fold
    })


fold_df = pd.DataFrame(
    fold_rows
)


print()
print("============================================================")
print("WITHIN-FOLD DETECTOR COMPARISON")
print("============================================================")

print(
    fold_df.to_string(
        index=False
    )
)


# ============================================================
# WIN COUNTS
# ============================================================

raw_auc_wins = int(
    (
        fold_df[
            "Raw_Delta_AUROC"
        ] > 0
    ).sum()
)

cal_auc_wins = int(
    (
        fold_df[
            "Calibrated_Delta_AUROC"
        ] > 0
    ).sum()
)

raw_ap_wins = int(
    (
        fold_df[
            "Raw_Delta_AUPRC"
        ] > 0
    ).sum()
)

cal_ap_wins = int(
    (
        fold_df[
            "Calibrated_Delta_AUPRC"
        ] > 0
    ).sum()
)


print()
print("============================================================")
print("WITHIN-FOLD WIN COUNTS")
print("============================================================")

print(
    "Raw uncertainty AUROC wins:",
    raw_auc_wins,
    "/",
    len(fold_df)
)

print(
    "Calibrated uncertainty AUROC wins:",
    cal_auc_wins,
    "/",
    len(fold_df)
)

print(
    "Raw uncertainty AUPRC wins:",
    raw_ap_wins,
    "/",
    len(fold_df)
)

print(
    "Calibrated uncertainty AUPRC wins:",
    cal_ap_wins,
    "/",
    len(fold_df)
)


# ============================================================
# TOP-K PRECISION
# ============================================================

precision_rows = []


for fraction in TOP_K_LEVELS:

    n_top = max(
        1,
        int(
            len(df)
            * fraction
        )
    )


    def precision_at_top(
        score
    ):

        order = np.argsort(
            score
        )[::-1]

        selected = order[
            :n_top
        ]

        return (
            y[selected]
            .mean()
        )


    precision_rows.append({

        "Top_Fraction":
            fraction,

        "OOD_Precision":
            precision_at_top(
                ood_score
            ),

        "Raw_Uncertainty_Precision":
            precision_at_top(
                raw_uncertainty
            ),

        "Calibrated_Uncertainty_Precision":
            precision_at_top(
                calibrated_uncertainty
            )
    })


precision_df = pd.DataFrame(
    precision_rows
)


print()
print("============================================================")
print("TOP-K PRECISION")
print("============================================================")

print(
    precision_df.to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

summary_rows = [

    {
        "Comparison":
            "Raw_Uncertainty_minus_OOD",

        "Metric":
            "AUROC",

        "OOD_Value":
            ood_auc,

        "Uncertainty_Value":
            raw_auc,

        "Difference":
            observed_raw_auc_diff,

        "CI_Low":
            raw_auc_ci[0],

        "CI_High":
            raw_auc_ci[1],

        "Permutation_p":
            raw_auc_p
    },

    {
        "Comparison":
            "Raw_Uncertainty_minus_OOD",

        "Metric":
            "AUPRC",

        "OOD_Value":
            ood_auprc,

        "Uncertainty_Value":
            raw_auprc,

        "Difference":
            observed_raw_auprc_diff,

        "CI_Low":
            raw_auprc_ci[0],

        "CI_High":
            raw_auprc_ci[1],

        "Permutation_p":
            raw_auprc_p
    },

    {
        "Comparison":
            "Calibrated_Uncertainty_minus_OOD",

        "Metric":
            "AUROC",

        "OOD_Value":
            ood_auc,

        "Uncertainty_Value":
            calibrated_auc,

        "Difference":
            observed_cal_auc_diff,

        "CI_Low":
            cal_auc_ci[0],

        "CI_High":
            cal_auc_ci[1],

        "Permutation_p":
            cal_auc_p
    },

    {
        "Comparison":
            "Calibrated_Uncertainty_minus_OOD",

        "Metric":
            "AUPRC",

        "OOD_Value":
            ood_auprc,

        "Uncertainty_Value":
            calibrated_auprc,

        "Difference":
            observed_cal_auprc_diff,

        "CI_Low":
            cal_auprc_ci[0],

        "CI_High":
            cal_auprc_ci[1],

        "Permutation_p":
            cal_auprc_p
    }
]


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE
# ============================================================

summary_df.to_csv(
    "YCEO_UncertaintyVsOOD_Significance.csv",
    index=False
)

fold_df.to_csv(
    "YCEO_UncertaintyVsOOD_Significance_ByFold.csv",
    index=False
)

precision_df.to_csv(
    "YCEO_UncertaintyVsOOD_Significance_TopK.csv",
    index=False
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report = []

report.append(
    "UNCERTAINTY VS OOD SIGNIFICANCE ANALYSIS"
)

report.append(
    ""
)

report.append(
    f"Observations: {len(df)}"
)

report.append(
    f"High-error threshold: {error_threshold:.6f}"
)

report.append(
    f"High-error prevalence: {y.mean() * 100:.3f}%"
)

report.append(
    ""
)

report.append(
    f"OOD AUROC: {ood_auc:.6f}"
)

report.append(
    f"Raw uncertainty AUROC: {raw_auc:.6f}"
)

report.append(
    f"Calibrated uncertainty AUROC: {calibrated_auc:.6f}"
)

report.append(
    ""
)

report.append(
    f"Raw uncertainty ΔAUROC: "
    f"{observed_raw_auc_diff:.6f}"
)

report.append(
    f"Raw uncertainty AUROC CI: "
    f"[{raw_auc_ci[0]:.6f}, {raw_auc_ci[1]:.6f}]"
)

report.append(
    f"Raw uncertainty AUROC permutation p: "
    f"{raw_auc_p:.6f}"
)

report.append(
    ""
)

report.append(
    f"Calibrated uncertainty ΔAUROC: "
    f"{observed_cal_auc_diff:.6f}"
)

report.append(
    f"Calibrated uncertainty AUROC CI: "
    f"[{cal_auc_ci[0]:.6f}, {cal_auc_ci[1]:.6f}]"
)

report.append(
    f"Calibrated uncertainty AUROC permutation p: "
    f"{cal_auc_p:.6f}"
)

report.append(
    ""
)

report.append(
    f"OOD AUPRC: {ood_auprc:.6f}"
)

report.append(
    f"Raw uncertainty AUPRC: {raw_auprc:.6f}"
)

report.append(
    f"Calibrated uncertainty AUPRC: "
    f"{calibrated_auprc:.6f}"
)

report.append(
    ""
)

report.append(
    f"Raw uncertainty ΔAUPRC: "
    f"{observed_raw_auprc_diff:.6f}"
)

report.append(
    f"Raw uncertainty AUPRC CI: "
    f"[{raw_auprc_ci[0]:.6f}, "
    f"{raw_auprc_ci[1]:.6f}]"
)

report.append(
    f"Raw uncertainty AUPRC permutation p: "
    f"{raw_auprc_p:.6f}"
)

report.append(
    ""
)

report.append(
    f"Calibrated uncertainty ΔAUPRC: "
    f"{observed_cal_auprc_diff:.6f}"
)

report.append(
    f"Calibrated uncertainty AUPRC CI: "
    f"[{cal_auprc_ci[0]:.6f}, "
    f"{cal_auprc_ci[1]:.6f}]"
)

report.append(
    f"Calibrated uncertainty AUPRC permutation p: "
    f"{cal_auprc_p:.6f}"
)

report.append(
    ""
)

report.append(
    f"OOD Spearman: {ood_spearman:.6f}"
)

report.append(
    f"Raw uncertainty Spearman: "
    f"{raw_spearman:.6f}"
)

report.append(
    f"Calibrated uncertainty Spearman: "
    f"{cal_spearman:.6f}"
)


with open(
    "YCEO_UncertaintyVsOOD_Significance_Report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_UncertaintyVsOOD_Significance.csv"
)

print(
    "YCEO_UncertaintyVsOOD_Significance_ByFold.csv"
)

print(
    "YCEO_UncertaintyVsOOD_Significance_TopK.csv"
)

print(
    "YCEO_UncertaintyVsOOD_Significance_Report.txt"
)

print()
print(
    "Paired detector significance analysis complete."
)