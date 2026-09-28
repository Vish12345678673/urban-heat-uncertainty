import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

OBSERVATION_FILE = (
    "YCEO_Uncertainty_ObservationResults.csv"
)

RANDOM_STATE = 42

# Internal calibration split.
CALIBRATION_FRACTION = 0.50

N_BINS = 10

Z_VALUES = {
    "80": 1.281551565545,
    "90": 1.644853626951,
    "95": 1.959963984540
}


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    OBSERVATION_FILE
)

required = [
    "Fold",
    "Observed",
    "Predicted",
    "Absolute_Error",
    "Squared_Error",
    "Predicted_Sigma"
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


print()
print("============================================================")
print("LEAKAGE-FREE UNCERTAINTY CALIBRATION")
print("============================================================")

print(
    "Observations:",
    len(df)
)

print(
    "Calibration fraction:",
    CALIBRATION_FRACTION
)


# ============================================================
# IMPORTANT DESIGN
# ============================================================
#
# The observation file contains OUTER TEST predictions.
#
# We must NOT simply fit a calibrator on all observations and
# report performance on the same observations.
#
# Instead, calibration is performed separately inside each
# outer fold using the first half of that fold's held-out
# observations as a calibration subset.
#
# The remaining half is the evaluation subset.
#
# This is not the final ideal nested calibration design, but it
# prevents direct calibrator/evaluation reuse within each fold.
# ============================================================


calibration_records = []

evaluation_records = []


for fold in sorted(
    df["Fold"].unique()
):

    fold_df = df[
        df["Fold"] == fold
    ].copy()


    rng = np.random.default_rng(
        RANDOM_STATE + int(fold)
    )


    indices = np.arange(
        len(fold_df)
    )

    rng.shuffle(
        indices
    )


    n_cal = int(
        len(fold_df)
        * CALIBRATION_FRACTION
    )


    calibration_idx = indices[
        :n_cal
    ]

    evaluation_idx = indices[
        n_cal:
    ]


    cal = fold_df.iloc[
        calibration_idx
    ].copy()

    evaluation = fold_df.iloc[
        evaluation_idx
    ].copy()


    calibration_records.append(
        cal
    )

    evaluation_records.append(
        evaluation
    )


calibration_df = pd.concat(
    calibration_records,
    ignore_index=True
)

evaluation_df = pd.concat(
    evaluation_records,
    ignore_index=True
)


# ============================================================
# CALIBRATION TARGET
# ============================================================
#
# We estimate the empirical relationship:
#
#       predicted sigma → absolute residual
#
# using isotonic regression.
#
# Isotonic regression is monotonic, so calibrated uncertainty
# cannot randomly decrease as raw uncertainty increases.
# ============================================================

iso = IsotonicRegression(
    increasing=True,
    out_of_bounds="clip"
)


iso.fit(
    calibration_df[
        "Predicted_Sigma"
    ],
    calibration_df[
        "Absolute_Error"
    ]
)


evaluation_df[
    "Calibrated_Risk"
] = iso.predict(
    evaluation_df[
        "Predicted_Sigma"
    ]
)


# ============================================================
# OPTIONAL SCALE CALIBRATION
# ============================================================
#
# Convert calibrated absolute-error risk into a sigma-like
# quantity using the Gaussian relation:
#
#     E|e| = sigma * sqrt(2/pi)
#
# Therefore:
#
#     sigma ≈ calibrated_risk / sqrt(2/pi)
#
# This is only a convenient scale transformation.
# ============================================================

evaluation_df[
    "Calibrated_Sigma"
] = (
    evaluation_df[
        "Calibrated_Risk"
    ]
    /
    np.sqrt(
        2.0 / np.pi
    )
)


evaluation_df[
    "Calibrated_Sigma"
] = np.maximum(
    evaluation_df[
        "Calibrated_Sigma"
    ],
    1e-6
)


# ============================================================
# COVERAGE FUNCTION
# ============================================================

def interval_metrics(
    data,
    z
):

    prediction = data[
        "Predicted"
    ].to_numpy()

    sigma = data[
        "Calibrated_Sigma"
    ].to_numpy()

    observed = data[
        "Observed"
    ].to_numpy()


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
# CALIBRATION RESULTS
# ============================================================

coverage_rows = []


for level, z in Z_VALUES.items():

    coverage, width = interval_metrics(
        evaluation_df,
        z
    )


    coverage_rows.append({

        "Nominal_Level":
            float(level) / 100.0,

        "Nominal_Percent":
            float(level),

        "Z":
            z,

        "Observed_Coverage":
            coverage,

        "Observed_Coverage_Percent":
            coverage * 100.0,

        "Mean_Interval_Width":
            width,

        "Absolute_Coverage_Error":
            abs(
                coverage
                -
                float(level) / 100.0
            )
    })


coverage_df = pd.DataFrame(
    coverage_rows
)


# ============================================================
# PRINT COVERAGE
# ============================================================

print()
print("============================================================")
print("PREDICTION INTERVAL CALIBRATION")
print("============================================================")

print(
    coverage_df.to_string(
        index=False
    )
)


# ============================================================
# RISK DISCRIMINATION
# ============================================================

pearson = np.corrcoef(
    evaluation_df[
        "Calibrated_Sigma"
    ],
    evaluation_df[
        "Absolute_Error"
    ]
)[0, 1]


spearman, spearman_p = spearmanr(
    evaluation_df[
        "Calibrated_Sigma"
    ],
    evaluation_df[
        "Absolute_Error"
    ]
)


print()
print("============================================================")
print("CALIBRATED UNCERTAINTY / ERROR ASSOCIATION")
print("============================================================")

print(
    "Pearson r:",
    round(
        pearson,
        6
    )
)

print(
    "Spearman r:",
    round(
        spearman,
        6
    )
)

print(
    "Spearman p:",
    spearman_p
)


# ============================================================
# UNCERTAINTY QUARTILES
# ============================================================

evaluation_df[
    "Calibrated_Uncertainty_Quartile"
] = pd.qcut(

    evaluation_df[
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

    evaluation_df

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
print("ERROR BY CALIBRATED UNCERTAINTY")
print("============================================================")

print(
    quartile_df.to_string(
        index=False
    )
)


# ============================================================
# Q4 / Q1
# ============================================================

q1 = quartile_df.iloc[0]

q4 = quartile_df.iloc[-1]


degradation = (

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
        degradation,
        4
    ),
    "%"
)


# ============================================================
# RISK-COVERAGE
# ============================================================

sorted_df = evaluation_df.sort_values(
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
# ERROR CONCENTRATION
# ============================================================

n_high = int(
    len(sorted_df) * 0.20
)


highest_risk = sorted_df.iloc[
    -n_high:
]


error_share = (
    highest_risk[
        "Absolute_Error"
    ].sum()
    /
    evaluation_df[
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
# FOLD COVERAGE
# ============================================================

fold_rows = []


for fold in sorted(
    evaluation_df["Fold"].unique()
):

    temp = evaluation_df[
        evaluation_df["Fold"] == fold
    ]


    fold_result = {
        "Fold":
            fold,

        "Observations":
            len(temp)
    }


    for level, z in Z_VALUES.items():

        coverage, width = interval_metrics(
            temp,
            z
        )


        fold_result[
            f"Coverage_{level}"
        ] = coverage

        fold_result[
            f"Width_{level}"
        ] = width


    fold_rows.append(
        fold_result
    )


fold_coverage_df = pd.DataFrame(
    fold_rows
)


print()
print("============================================================")
print("FOLD-LEVEL COVERAGE")
print("============================================================")

print(
    fold_coverage_df.to_string(
        index=False
    )
)


# ============================================================
# CALIBRATION SUMMARY
# ============================================================

calibration_summary = pd.DataFrame([{

    "Calibration_Observations":
        len(calibration_df),

    "Evaluation_Observations":
        len(evaluation_df),

    "Raw_Sigma_Mean":
        evaluation_df[
            "Predicted_Sigma"
        ].mean(),

    "Calibrated_Sigma_Mean":
        evaluation_df[
            "Calibrated_Sigma"
        ].mean(),

    "Pearson_Uncertainty_Error":
        pearson,

    "Spearman_Uncertainty_Error":
        spearman,

    "Spearman_p":
        spearman_p,

    "Q4_Q1_MAE_Degradation":
        degradation,

    "High_Risk_20_Error_Share":
        error_share
}])


# ============================================================
# SAVE
# ============================================================

evaluation_df.to_csv(
    "YCEO_Uncertainty_Calibrated_Observations.csv",
    index=False
)

coverage_df.to_csv(
    "YCEO_Uncertainty_Calibration_Coverage.csv",
    index=False
)

quartile_df.to_csv(
    "YCEO_Uncertainty_Calibrated_Quartiles.csv",
    index=False
)

risk_df.to_csv(
    "YCEO_Uncertainty_Calibrated_RiskCoverage.csv",
    index=False
)

fold_coverage_df.to_csv(
    "YCEO_Uncertainty_Calibrated_FoldCoverage.csv",
    index=False
)

calibration_summary.to_csv(
    "YCEO_Uncertainty_Calibration_Summary.csv",
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
    "YCEO_Uncertainty_Calibrated_Observations.csv"
)

print(
    "YCEO_Uncertainty_Calibration_Coverage.csv"
)

print(
    "YCEO_Uncertainty_Calibrated_Quartiles.csv"
)

print(
    "YCEO_Uncertainty_Calibrated_RiskCoverage.csv"
)

print(
    "YCEO_Uncertainty_Calibrated_FoldCoverage.csv"
)

print(
    "YCEO_Uncertainty_Calibration_Summary.csv"
)

print()
print(
    "Uncertainty calibration experiment complete."
)