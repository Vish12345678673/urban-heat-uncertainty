import pandas as pd
import numpy as np

from scipy.stats import spearmanr, pearsonr


# ============================================================
# LOAD V2 OBSERVATION RESULTS
# ============================================================

df = pd.read_csv(
    "YCEO_DistributionShiftV2_ObservationResults.csv"
)


print()
print("==============================================")
print("WITHIN-FOLD DISTRIBUTION-SHIFT ANALYSIS")
print("==============================================")

print(
    "Observations:",
    len(df)
)


# ============================================================
# OVERALL CORRELATION
# ============================================================

pearson_overall, pearson_p_overall = pearsonr(
    df["Shift_Distance"],
    df["Absolute_Error"]
)

spearman_overall, spearman_p_overall = spearmanr(
    df["Shift_Distance"],
    df["Absolute_Error"]
)


print()
print("Overall association:")

print(
    "Pearson r =",
    round(pearson_overall, 6),
    "p =",
    pearson_p_overall
)

print(
    "Spearman r =",
    round(spearman_overall, 6),
    "p =",
    spearman_p_overall
)


# ============================================================
# WITHIN-FOLD ANALYSIS
# ============================================================

fold_results = []


for fold in sorted(
    df["Fold"].unique()
):

    fold_df = df[
        df["Fold"] == fold
    ].copy()


    if len(fold_df) < 3:

        continue


    pearson_r, pearson_p = pearsonr(
        fold_df["Shift_Distance"],
        fold_df["Absolute_Error"]
    )

    spearman_r, spearman_p = spearmanr(
        fold_df["Shift_Distance"],
        fold_df["Absolute_Error"]
    )


    fold_results.append({

        "Fold": fold,

        "Observations": len(fold_df),

        "Pearson_r": pearson_r,

        "Pearson_p": pearson_p,

        "Spearman_r": spearman_r,

        "Spearman_p": spearman_p,

        "Mean_Shift_Distance":
            fold_df["Shift_Distance"].mean(),

        "Mean_Absolute_Error":
            fold_df["Absolute_Error"].mean()
    })


fold_results = pd.DataFrame(
    fold_results
)


# ============================================================
# PRINT FOLD RESULTS
# ============================================================

print()
print("==============================================")
print("WITHIN-FOLD CORRELATIONS")
print("==============================================")

print(
    fold_results.to_string(
        index=False
    )
)


# ============================================================
# COUNT DIRECTIONAL CONSISTENCY
# ============================================================

positive_folds = (
    fold_results["Spearman_r"] > 0
).sum()

negative_folds = (
    fold_results["Spearman_r"] < 0
).sum()


print()
print("==============================================")
print("DIRECTIONAL CONSISTENCY")
print("==============================================")

print(
    "Positive Spearman folds:",
    positive_folds
)

print(
    "Negative Spearman folds:",
    negative_folds
)


# ============================================================
# Q1-Q4 ANALYSIS WITHIN EACH FOLD
# ============================================================

quartile_results = []


for fold in sorted(
    df["Fold"].unique()
):

    fold_df = df[
        df["Fold"] == fold
    ].copy()


    try:

        fold_df["Local_Quartile"] = pd.qcut(
            fold_df["Shift_Distance"],
            q=4,
            labels=[
                "Q1 - closest",
                "Q2",
                "Q3",
                "Q4 - farthest"
            ],
            duplicates="drop"
        )

    except ValueError:

        continue


    summary = (
        fold_df
        .groupby(
            "Local_Quartile",
            observed=True
        )
        .agg(
            Observations=(
                "Code",
                "count"
            ),

            Mean_Shift_Distance=(
                "Shift_Distance",
                "mean"
            ),

            Mean_Absolute_Error=(
                "Absolute_Error",
                "mean"
            )
        )
        .reset_index()
    )


    summary["Fold"] = fold

    quartile_results.append(
        summary
    )


quartile_df = pd.concat(
    quartile_results,
    ignore_index=True
)


# ============================================================
# PRINT QUARTILE RESULTS
# ============================================================

print()
print("==============================================")
print("WITHIN-FOLD SHIFT QUARTILES")
print("==============================================")

print(
    quartile_df[
        [
            "Fold",
            "Local_Quartile",
            "Observations",
            "Mean_Shift_Distance",
            "Mean_Absolute_Error"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Q4 - Q1 ERROR DEGRADATION BY FOLD
# ============================================================

degradation = []


for fold in sorted(
    quartile_df["Fold"].unique()
):

    temp = quartile_df[
        quartile_df["Fold"] == fold
    ]


    q1 = temp[
        temp["Local_Quartile"]
        == "Q1 - closest"
    ]

    q4 = temp[
        temp["Local_Quartile"]
        == "Q4 - farthest"
    ]


    if (
        len(q1) == 1
        and
        len(q4) == 1
    ):

        q1_error = (
            q1["Mean_Absolute_Error"]
            .iloc[0]
        )

        q4_error = (
            q4["Mean_Absolute_Error"]
            .iloc[0]
        )


        degradation_percent = (
            (q4_error - q1_error)
            / q1_error
            * 100
        )


        degradation.append({

            "Fold": fold,

            "Q1_MAE": q1_error,

            "Q4_MAE": q4_error,

            "Q4_Q1_Degradation_percent":
                degradation_percent
        })


degradation_df = pd.DataFrame(
    degradation
)


# ============================================================
# PRINT DEGRADATION
# ============================================================

print()
print("==============================================")
print("Q4 VS Q1 ERROR DEGRADATION")
print("==============================================")

print(
    degradation_df.to_string(
        index=False
    )
)


# ============================================================
# SUMMARY ACROSS FOLDS
# ============================================================

print()
print("==============================================")
print("DEGRADATION SUMMARY")
print("==============================================")


if len(degradation_df) > 0:

    mean_degradation = (
        degradation_df[
            "Q4_Q1_Degradation_percent"
        ].mean()
    )

    sd_degradation = (
        degradation_df[
            "Q4_Q1_Degradation_percent"
        ].std(
            ddof=1
        )
    )

    print(
        "Mean Q4-Q1 degradation:",
        round(
            mean_degradation,
            4
        ),
        "%"
    )

    print(
        "SD:",
        round(
            sd_degradation,
            4
        ),
        "%"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

fold_results.to_csv(
    "YCEO_DistributionShift_WithinFold_Correlations.csv",
    index=False
)

quartile_df.to_csv(
    "YCEO_DistributionShift_WithinFold_Quartiles.csv",
    index=False
)

degradation_df.to_csv(
    "YCEO_DistributionShift_WithinFold_Degradation.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================================")
print("FILES CREATED")
print("==============================================")

print(
    "YCEO_DistributionShift_WithinFold_Correlations.csv"
)

print(
    "YCEO_DistributionShift_WithinFold_Quartiles.csv"
)

print(
    "YCEO_DistributionShift_WithinFold_Degradation.csv"
)

print()
print(
    "Within-fold distribution-shift analysis complete."
)