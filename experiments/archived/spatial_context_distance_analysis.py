import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr


# ============================================================
# LOAD RESULTS
# ============================================================

FILE = "YCEO_SpatialContext_Observation_Results.csv"

df = pd.read_csv(FILE)

print()
print("============================================================")
print("SPATIAL CONTEXT vs TRAINING-NEIGHBOR DISTANCE")
print("============================================================")

print(
    "Observations:",
    len(df)
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required = [
    "Fold",
    "Baseline_Absolute_Error",
    "SpatialContext_Absolute_Error",
    "Mean_Neighbor_Distance_km"
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
# BENEFIT OF SPATIAL CONTEXT
# ============================================================

df["MAE_Improvement"] = (
    df["Baseline_Absolute_Error"]
    -
    df["SpatialContext_Absolute_Error"]
)


# Positive = spatial context improves prediction
# Negative = spatial context hurts prediction


# ============================================================
# OVERALL SUMMARY
# ============================================================

print()
print("============================================================")
print("OVERALL EFFECT")
print("============================================================")

print(
    "Mean baseline absolute error:",
    round(
        df["Baseline_Absolute_Error"].mean(),
        6
    )
)

print(
    "Mean spatial-context absolute error:",
    round(
        df["SpatialContext_Absolute_Error"].mean(),
        6
    )
)

print(
    "Mean improvement:",
    round(
        df["MAE_Improvement"].mean(),
        6
    )
)


# ============================================================
# CORRELATION WITH NEIGHBOR DISTANCE
# ============================================================

pearson_r, pearson_p = pearsonr(
    df["Mean_Neighbor_Distance_km"],
    df["MAE_Improvement"]
)

spearman_r, spearman_p = spearmanr(
    df["Mean_Neighbor_Distance_km"],
    df["MAE_Improvement"]
)


print()
print("============================================================")
print("DISTANCE vs SPATIAL-CONTEXT BENEFIT")
print("============================================================")

print(
    "Pearson r :",
    round(
        pearson_r,
        6
    )
)

print(
    "Pearson p :",
    pearson_p
)

print(
    "Spearman r:",
    round(
        spearman_r,
        6
    )
)

print(
    "Spearman p:",
    spearman_p
)


# ============================================================
# QUARTILES OF NEIGHBOR DISTANCE
# ============================================================

df["Distance_Quartile"] = pd.qcut(

    df["Mean_Neighbor_Distance_km"],

    q=4,

    labels=[
        "Q1 - closest",
        "Q2",
        "Q3",
        "Q4 - farthest"
    ],

    duplicates="drop"
)


distance_summary = (
    df
    .groupby(
        "Distance_Quartile",
        observed=True
    )
    .agg(

        Observations=(
            "Fold",
            "count"
        ),

        Mean_Neighbor_Distance_km=(
            "Mean_Neighbor_Distance_km",
            "mean"
        ),

        Baseline_MAE=(
            "Baseline_Absolute_Error",
            "mean"
        ),

        SpatialContext_MAE=(
            "SpatialContext_Absolute_Error",
            "mean"
        ),

        Mean_Improvement=(
            "MAE_Improvement",
            "mean"
        ),

        Median_Improvement=(
            "MAE_Improvement",
            "median"
        )
    )
    .reset_index()
)


print()
print("============================================================")
print("SPATIAL-CONTEXT BENEFIT BY NEIGHBOR DISTANCE")
print("============================================================")

print(
    distance_summary.to_string(
        index=False
    )
)


# ============================================================
# BENEFIT BY FOLD
# ============================================================

fold_summary = (
    df
    .groupby(
        "Fold"
    )
    .agg(

        Observations=(
            "Fold",
            "count"
        ),

        Mean_Neighbor_Distance_km=(
            "Mean_Neighbor_Distance_km",
            "mean"
        ),

        Baseline_MAE=(
            "Baseline_Absolute_Error",
            "mean"
        ),

        SpatialContext_MAE=(
            "SpatialContext_Absolute_Error",
            "mean"
        ),

        Mean_Improvement=(
            "MAE_Improvement",
            "mean"
        )
    )
    .reset_index()
)


print()
print("============================================================")
print("BENEFIT BY SPATIAL FOLD")
print("============================================================")

print(
    fold_summary.to_string(
        index=False
    )
)


# ============================================================
# FRACTION OF OBSERVATIONS IMPROVED
# ============================================================

improved_fraction = (
    df["MAE_Improvement"] > 0
).mean()

worsened_fraction = (
    df["MAE_Improvement"] < 0
).mean()

equal_fraction = (
    df["MAE_Improvement"] == 0
).mean()


print()
print("============================================================")
print("OBSERVATION-LEVEL BENEFIT")
print("============================================================")

print(
    "Improved:",
    round(
        improved_fraction * 100,
        2
    ),
    "%"
)

print(
    "Worsened:",
    round(
        worsened_fraction * 100,
        2
    ),
    "%"
)

print(
    "Unchanged:",
    round(
        equal_fraction * 100,
        2
    ),
    "%"
)


# ============================================================
# BY DISTANCE QUARTILE: IMPROVEMENT RATE
# ============================================================

improvement_rate = (
    df
    .groupby(
        "Distance_Quartile",
        observed=True
    )
    .agg(

        Improvement_Rate=(
            "MAE_Improvement",
            lambda x: (
                x > 0
            ).mean()
        )
    )
    .reset_index()
)

improvement_rate[
    "Improvement_Rate"
] *= 100


print()
print("============================================================")
print("IMPROVEMENT RATE BY DISTANCE")
print("============================================================")

print(
    improvement_rate.to_string(
        index=False
    )
)


# ============================================================
# SAVE OBSERVATION RESULTS
# ============================================================

df.to_csv(
    "YCEO_SpatialContext_Distance_ObservationResults.csv",
    index=False
)


distance_summary.to_csv(
    "YCEO_SpatialContext_Distance_Quartiles.csv",
    index=False
)


fold_summary.to_csv(
    "YCEO_SpatialContext_Distance_FoldSummary.csv",
    index=False
)


improvement_rate.to_csv(
    "YCEO_SpatialContext_Distance_ImprovementRate.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "Pearson_r":
        pearson_r,

    "Pearson_p":
        pearson_p,

    "Spearman_r":
        spearman_r,

    "Spearman_p":
        spearman_p,

    "Mean_Improvement":
        df["MAE_Improvement"].mean(),

    "Improvement_Rate_percent":
        improved_fraction * 100,

    "Worsened_Rate_percent":
        worsened_fraction * 100
}])


summary.to_csv(
    "YCEO_SpatialContext_Distance_Summary.csv",
    index=False
)


print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_SpatialContext_Distance_ObservationResults.csv"
)

print(
    "YCEO_SpatialContext_Distance_Quartiles.csv"
)

print(
    "YCEO_SpatialContext_Distance_FoldSummary.csv"
)

print(
    "YCEO_SpatialContext_Distance_ImprovementRate.csv"
)

print(
    "YCEO_SpatialContext_Distance_Summary.csv"
)

print()
print(
    "Spatial-context distance analysis complete."
)