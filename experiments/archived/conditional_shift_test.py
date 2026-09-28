import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# ============================================================
# LOAD LEAKAGE-FREE OBSERVATION RESULTS
# ============================================================

df = pd.read_csv(
    "YCEO_DistributionShiftV2_ObservationResults.csv"
)

print()
print("==============================================")
print("CONDITIONAL-SHIFT TEST")
print("==============================================")

print(
    "Observations:",
    len(df)
)


# ============================================================
# CHECK DATA
# ============================================================

required = [
    "Fold",
    "Shift_Distance",
    "Absolute_Error"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing columns: " + str(missing)
    )


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
# CENTER SHIFT DISTANCE
# ============================================================
#
# Centering makes the interaction model easier to interpret.
# ============================================================

df["Shift_Centered"] = (
    df["Shift_Distance"]
    - df["Shift_Distance"].mean()
)


# ============================================================
# MODEL 1
#
# ERROR ~ SHIFT + FOLD
# ============================================================

model1 = smf.ols(
    "Absolute_Error ~ Shift_Centered + C(Fold)",
    data=df
).fit(
    cov_type="HC3"
)


# ============================================================
# MODEL 2
#
# ERROR ~ SHIFT * FOLD
#
# This allows the shift-error relationship to differ
# between geographic folds.
# ============================================================

model2 = smf.ols(
    "Absolute_Error ~ Shift_Centered * C(Fold)",
    data=df
).fit(
    cov_type="HC3"
)


# ============================================================
# MODEL FIT
# ============================================================

print()
print("==============================================")
print("MODEL FIT")
print("==============================================")

print(
    "Model 1 R²:",
    round(
        model1.rsquared,
        6
    )
)

print(
    "Model 2 R²:",
    round(
        model2.rsquared,
        6
    )
)

print(
    "R² improvement:",
    round(
        model2.rsquared
        - model1.rsquared,
        6
    )
)


# ============================================================
# F-TEST OF INTERACTIONS
# ============================================================

interaction_terms = [
    term
    for term in model2.params.index
    if "Shift_Centered:C(Fold)" in term
]


print()
print("==============================================")
print("SHIFT × FOLD INTERACTIONS")
print("==============================================")


interaction_rows = []


for term in interaction_terms:

    coefficient = model2.params[
        term
    ]

    p_value = model2.pvalues[
        term
    ]

    ci_low, ci_high = (
        model2.conf_int()
        .loc[term]
    )

    print()
    print(term)

    print(
        "Coefficient:",
        round(
            coefficient,
            6
        )
    )

    print(
        "95% CI:",
        (
            round(
                ci_low,
                6
            ),
            round(
                ci_high,
                6
            )
        )
    )

    print(
        "p-value:",
        p_value
    )

    interaction_rows.append({

        "Term": term,

        "Coefficient": coefficient,

        "CI_Low": ci_low,

        "CI_High": ci_high,

        "p_value": p_value
    })


# ============================================================
# JOINT F-TEST
# ============================================================

if len(interaction_terms) > 0:

    joint_test = model2.f_test(
        " = 0, ".join(
            interaction_terms
        )
        + " = 0"
    )

    joint_p = float(
        joint_test.pvalue
    )

else:

    joint_p = np.nan


print()
print("==============================================")
print("JOINT INTERACTION TEST")
print("==============================================")

print(
    "Interaction terms:",
    len(interaction_terms)
)

print(
    "Joint p-value:",
    joint_p
)


# ============================================================
# FOLD-SPECIFIC SHIFT SLOPES
# ============================================================
#
# Calculate a separate simple regression within each fold.
# ============================================================

fold_slopes = []


print()
print("==============================================")
print("FOLD-SPECIFIC SHIFT SLOPES")
print("==============================================")


for fold in sorted(
    df["Fold"].unique()
):

    fold_df = df[
        df["Fold"] == fold
    ]

    fold_model = smf.ols(
        "Absolute_Error ~ Shift_Centered",
        data=fold_df
    ).fit(
        cov_type="HC3"
    )

    slope = fold_model.params[
        "Shift_Centered"
    ]

    p_value = fold_model.pvalues[
        "Shift_Centered"
    ]

    ci_low, ci_high = (
        fold_model.conf_int()
        .loc[
            "Shift_Centered"
        ]
    )

    print()
    print(
        "Fold",
        fold
    )

    print(
        "Slope:",
        round(
            slope,
            6
        )
    )

    print(
        "95% CI:",
        (
            round(
                ci_low,
                6
            ),
            round(
                ci_high,
                6
            )
        )
    )

    print(
        "p-value:",
        p_value
    )

    fold_slopes.append({

        "Fold": fold,

        "Slope": slope,

        "CI_Low": ci_low,

        "CI_High": ci_high,

        "p_value": p_value
    })


fold_slopes_df = pd.DataFrame(
    fold_slopes
)


# ============================================================
# TEST WHETHER SLOPES ARE ALL POSITIVE
# ============================================================

positive_slopes = (
    fold_slopes_df["Slope"] > 0
).sum()

significant_positive = (
    (
        fold_slopes_df["Slope"] > 0
    )
    &
    (
        fold_slopes_df["p_value"] < 0.05
    )
).sum()


print()
print("==============================================")
print("SLOPE CONSISTENCY")
print("==============================================")

print(
    "Positive fold slopes:",
    positive_slopes,
    "/",
    len(fold_slopes_df)
)

print(
    "Significant positive slopes:",
    significant_positive,
    "/",
    len(fold_slopes_df)
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("==============================================")
print("INTERPRETATION")
print("==============================================")


if (
    joint_p < 0.05
):

    interpretation = (
        "The shift-error relationship differs "
        "significantly across spatial folds. "
        "This supports heterogeneous domain behavior "
        "beyond a single pooled shift effect."
    )

else:

    interpretation = (
        "No statistically significant evidence was "
        "found that the shift-error relationship differs "
        "across spatial folds."
    )


print(
    interpretation
)


# ============================================================
# SAVE RESULTS
# ============================================================

interaction_df = pd.DataFrame(
    interaction_rows
)

interaction_df.to_csv(
    "YCEO_ConditionalShift_InteractionResults.csv",
    index=False
)

fold_slopes_df.to_csv(
    "YCEO_ConditionalShift_FoldSlopes.csv",
    index=False
)

summary = pd.DataFrame([{

    "Model1_R2": model1.rsquared,

    "Model2_R2": model2.rsquared,

    "R2_Improvement":
        model2.rsquared
        - model1.rsquared,

    "Joint_Interaction_p":
        joint_p,

    "Positive_Fold_Slopes":
        positive_slopes,

    "Significant_Positive_Fold_Slopes":
        significant_positive,

    "Interpretation":
        interpretation
}])

summary.to_csv(
    "YCEO_ConditionalShift_Summary.csv",
    index=False
)


# ============================================================
# FULL MODEL SUMMARIES
# ============================================================

with open(
    "YCEO_ConditionalShift_Regression_Summaries.txt",
    "w"
) as f:

    f.write(
        "MODEL 1: ERROR ~ SHIFT + FOLD\n\n"
    )

    f.write(
        model1.summary().as_text()
    )

    f.write(
        "\n\nMODEL 2: ERROR ~ SHIFT * FOLD\n\n"
    )

    f.write(
        model2.summary().as_text()
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================================")
print("FILES CREATED")
print("==============================================")

print(
    "YCEO_ConditionalShift_InteractionResults.csv"
)

print(
    "YCEO_ConditionalShift_FoldSlopes.csv"
)

print(
    "YCEO_ConditionalShift_Summary.csv"
)

print(
    "YCEO_ConditionalShift_Regression_Summaries.txt"
)

print()
print(
    "Conditional-shift analysis complete."
)