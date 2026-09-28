import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr


# =====================================================
# LOAD FILES
# =====================================================

shap_df = pd.read_csv(
    "YCEO_UHI_Final_SHAP_Values.csv"
)

original = pd.read_csv(
    "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"
)


# =====================================================
# FEATURES
# =====================================================

features = [
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


# =====================================================
# IMPORTANT:
# SHAP FILE ALREADY CONTAINS THE SHAP COLUMNS.
# ORIGINAL DATA CONTAINS THE ACTUAL FEATURE VALUES.
# We therefore explicitly rename the actual values.
# =====================================================

actual = original[
    ["Code"] + features
].copy()

actual = actual.rename(
    columns={
        feature: feature + "_value"
        for feature in features
    }
)


# =====================================================
# MATCH ACTUAL VALUES TO SHAP VALUES
# =====================================================

df = shap_df.merge(
    actual,
    on="Code",
    how="left"
)


print()
print("Matched SHAP observations:", len(df))


# =====================================================
# ANALYZE DIRECTION
# =====================================================

results = []


for feature in features:

    shap_values = df[feature]

    actual_values = df[
        feature + "_value"
    ]

    valid = (
        shap_values.notna()
        &
        actual_values.notna()
    )

    s = shap_values[valid]
    x = actual_values[valid]

    pearson_r, pearson_p = pearsonr(
        x,
        s
    )

    spearman_r, spearman_p = spearmanr(
        x,
        s
    )

    mean_shap = s.mean()

    mean_abs_shap = np.abs(
        s
    ).mean()

    positive_percent = (
        (s > 0).mean()
        * 100
    )

    negative_percent = (
        (s < 0).mean()
        * 100
    )

    results.append({

        "Feature": feature,

        "Mean_SHAP":
            mean_shap,

        "Mean_Absolute_SHAP":
            mean_abs_shap,

        "Pearson_r":
            pearson_r,

        "Pearson_p":
            pearson_p,

        "Spearman_r":
            spearman_r,

        "Spearman_p":
            spearman_p,

        "Positive_SHAP_percent":
            positive_percent,

        "Negative_SHAP_percent":
            negative_percent

    })


# =====================================================
# RESULTS TABLE
# =====================================================

results = pd.DataFrame(
    results
)

results = results.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


# =====================================================
# DIRECTION LABEL
# =====================================================

def classify_direction(r):

    if r >= 0.30:
        return "Positive"

    elif r <= -0.30:
        return "Negative"

    else:
        return "Weak / Mixed / Nonlinear"


results["Overall_Direction"] = (
    results["Spearman_r"]
    .apply(classify_direction)
)


# =====================================================
# SAVE
# =====================================================

output_file = (
    "YCEO_UHI_SHAP_Direction_Analysis.csv"
)

results.to_csv(
    output_file,
    index=False
)


# =====================================================
# PRINT
# =====================================================

print()
print("==========================================")
print("SHAP DIRECTION ANALYSIS")
print("==========================================")
print()

print(
    results.to_string(
        index=False
    )
)


print()
print("==========================================")
print("INTERPRETATION")
print("==========================================")


for _, row in results.iterrows():

    print()
    print(
        row["Feature"]
    )

    print(
        "Spearman r:",
        round(
            row["Spearman_r"],
            4
        )
    )

    print(
        "Mean SHAP:",
        round(
            row["Mean_SHAP"],
            4
        )
    )

    print(
        "Positive SHAP:",
        round(
            row["Positive_SHAP_percent"],
            1
        ),
        "%"
    )

    print(
        "Negative SHAP:",
        round(
            row["Negative_SHAP_percent"],
            1
        ),
        "%"
    )

    print(
        "Direction:",
        row["Overall_Direction"]
    )


print()
print(
    "Saved:",
    output_file
)