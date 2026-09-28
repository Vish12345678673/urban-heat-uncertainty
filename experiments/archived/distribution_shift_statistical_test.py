import pandas as pd
import numpy as np
import subprocess
import sys
import importlib.util

# ============================================================
# INSTALL STATSMODELS IF NEEDED
# ============================================================

if importlib.util.find_spec("statsmodels") is None:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "statsmodels",
        "-q"
    ])

import statsmodels.formula.api as smf


# ============================================================
# LOAD OBSERVATION-LEVEL RESULTS
# ============================================================

df = pd.read_csv(
    "YCEO_DistributionShiftV2_ObservationResults.csv"
)

print()
print("==============================================")
print("FORMAL DISTRIBUTION-SHIFT STATISTICAL TEST")
print("==============================================")

print(
    "Observations:",
    len(df)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "Shift_Distance",
    "Absolute_Error",
    "Fold"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns: " + str(missing)
    )


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

df = df[
    np.isfinite(
        df["Shift_Distance"]
    )
    &
    np.isfinite(
        df["Absolute_Error"]
    )
].copy()


# ============================================================
# STANDARDIZE SHIFT DISTANCE
# ============================================================

df["Shift_Distance_Z"] = (
    df["Shift_Distance"]
    - df["Shift_Distance"].mean()
) / df["Shift_Distance"].std()


# ============================================================
# MODEL 1
# ERROR ~ SHIFT
# ============================================================

model1 = smf.ols(
    "Absolute_Error ~ Shift_Distance",
    data=df
).fit(
    cov_type="HC3"
)


# ============================================================
# MODEL 2
# ERROR ~ SHIFT + SPATIAL FOLD
# ============================================================

model2 = smf.ols(
    "Absolute_Error ~ Shift_Distance + C(Fold)",
    data=df
).fit(
    cov_type="HC3"
)


# ============================================================
# MODEL 3
# STANDARDIZED SHIFT + SPATIAL FOLD
# ============================================================

model3 = smf.ols(
    "Absolute_Error ~ Shift_Distance_Z + C(Fold)",
    data=df
).fit(
    cov_type="HC3"
)


# ============================================================
# EXTRACT SHIFT EFFECT
# ============================================================

coef = model2.params[
    "Shift_Distance"
]

p_value = model2.pvalues[
    "Shift_Distance"
]

ci_low, ci_high = model2.conf_int().loc[
    "Shift_Distance"
]


# ============================================================
# MODEL FIT
# ============================================================

r2_model1 = model1.rsquared
r2_model2 = model2.rsquared
r2_model3 = model3.rsquared


# ============================================================
# CORRELATION
# ============================================================

pearson = df[
    [
        "Shift_Distance",
        "Absolute_Error"
    ]
].corr(
    method="pearson"
).iloc[0, 1]

spearman = df[
    [
        "Shift_Distance",
        "Absolute_Error"
    ]
].corr(
    method="spearman"
).iloc[0, 1]


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("==============================================")
print("OVERALL ASSOCIATION")
print("==============================================")

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


print()
print("==============================================")
print("MODEL 1: ERROR ~ SHIFT")
print("==============================================")

print(
    "R²:",
    round(
        r2_model1,
        6
    )
)

print(
    "Shift coefficient:",
    round(
        model1.params["Shift_Distance"],
        6
    )
)

print(
    "p-value:",
    model1.pvalues["Shift_Distance"]
)


print()
print("==============================================")
print("MODEL 2: ERROR ~ SHIFT + FOLD")
print("==============================================")

print(
    "R²:",
    round(
        r2_model2,
        6
    )
)

print(
    "Shift coefficient:",
    round(
        coef,
        6
    )
)

print(
    "95% CI:",
    (
        round(ci_low, 6),
        round(ci_high, 6)
    )
)

print(
    "p-value:",
    p_value
)


print()
print("==============================================")
print("MODEL 3: STANDARDIZED SHIFT + FOLD")
print("==============================================")

print(
    "R²:",
    round(
        r2_model3,
        6
    )
)

print(
    "Standardized shift coefficient:",
    round(
        model3.params[
            "Shift_Distance_Z"
        ],
        6
    )
)

print(
    "p-value:",
    model3.pvalues[
        "Shift_Distance_Z"
    ]
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("==============================================")
print("INTERPRETATION")
print("==============================================")

if (
    coef > 0
    and
    p_value < 0.05
):

    print(
        "Greater predictor-distribution shift is "
        "significantly associated with larger "
        "absolute prediction error after accounting "
        "for spatial fold."
    )

elif p_value < 0.05:

    print(
        "Distribution-shift distance is significantly "
        "associated with absolute prediction error "
        "after accounting for spatial fold, but the "
        "estimated association is negative."
    )

else:

    print(
        "The shift-error association is not "
        "statistically significant after accounting "
        "for spatial fold."
    )


# ============================================================
# SAVE STATISTICAL RESULTS
# ============================================================

stat_results = pd.DataFrame({

    "Analysis": [
        "Pearson correlation",
        "Spearman correlation",
        "Model 1 R2",
        "Model 1 shift coefficient",
        "Model 1 shift p-value",
        "Model 2 R2",
        "Model 2 shift coefficient",
        "Model 2 shift CI low",
        "Model 2 shift CI high",
        "Model 2 shift p-value",
        "Model 3 R2",
        "Model 3 standardized shift coefficient",
        "Model 3 standardized shift p-value"
    ],

    "Value": [

        pearson,

        spearman,

        r2_model1,

        model1.params[
            "Shift_Distance"
        ],

        model1.pvalues[
            "Shift_Distance"
        ],

        r2_model2,

        coef,

        ci_low,

        ci_high,

        p_value,

        r2_model3,

        model3.params[
            "Shift_Distance_Z"
        ],

        model3.pvalues[
            "Shift_Distance_Z"
        ]
    ]
})


stat_results.to_csv(
    "YCEO_DistributionShift_Statistical_Test.csv",
    index=False
)


# ============================================================
# SAVE FULL MODEL SUMMARIES
# ============================================================

with open(
    "YCEO_DistributionShift_Regression_Summaries.txt",
    "w"
) as f:

    f.write(
        "MODEL 1\n"
    )

    f.write(
        model1.summary().as_text()
    )

    f.write(
        "\n\nMODEL 2\n"
    )

    f.write(
        model2.summary().as_text()
    )

    f.write(
        "\n\nMODEL 3\n"
    )

    f.write(
        model3.summary().as_text()
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================================")
print("FILES CREATED")
print("==============================================")

print(
    "YCEO_DistributionShift_Statistical_Test.csv"
)

print(
    "YCEO_DistributionShift_Regression_Summaries.txt"
)

print()
print(
    "Formal distribution-shift test complete."
)