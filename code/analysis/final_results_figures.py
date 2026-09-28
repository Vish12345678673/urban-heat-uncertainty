import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================

out = Path("Paper_Figures")
out.mkdir(exist_ok=True)


# =====================================================
# 1. MODEL COMPARISON FIGURE
# =====================================================

models = pd.read_csv(
    "YCEO_UHI_Model_Comparison.csv"
)

models = models.sort_values(
    "Spatial_R2",
    ascending=True
)

plt.figure(figsize=(8, 5))

plt.barh(
    models["Model"],
    models["Spatial_R2"]
)

plt.xlabel("Spatial R²")
plt.ylabel("Model")
plt.title(
    "Model Performance under Spatial Validation"
)

plt.xlim(
    0,
    max(models["Spatial_R2"]) * 1.15
)

plt.tight_layout()

plt.savefig(
    out / "Figure_1_Model_Comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 2. CROSS-VALIDATION FIGURE
# =====================================================

cv = pd.read_csv(
    "YCEO_UHI_5Fold_Spatial_CV.csv"
)

plt.figure(figsize=(8, 5))

plt.plot(
    cv["Fold"],
    cv["R2"],
    marker="o",
    linewidth=1.8
)

mean_r2 = cv["R2"].mean()

plt.axhline(
    mean_r2,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel("Spatial Fold")
plt.ylabel("R²")
plt.title(
    "Five-Fold Spatial Cross-Validation"
)

plt.xticks(
    cv["Fold"]
)

plt.tight_layout()

plt.savefig(
    out / "Figure_2_Spatial_CV.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 3. SHAP GLOBAL IMPORTANCE
# =====================================================

shap = pd.read_csv(
    "YCEO_UHI_Final_SHAP_Importance.csv"
)

shap = shap.sort_values(
    "Mean_Absolute_SHAP",
    ascending=True
)

plt.figure(figsize=(8, 6))

plt.barh(
    shap["Feature"],
    shap["Mean_Absolute_SHAP"]
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Predictor"
)

plt.title(
    "Global Feature Importance from SHAP"
)

plt.tight_layout()

plt.savefig(
    out / "Figure_3_SHAP_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 4. OBSERVED VS PREDICTED
# =====================================================

pred = pd.read_csv(
    "YCEO_UHI_Final_Predictions.csv"
)

plt.figure(figsize=(7, 7))

plt.scatter(
    pred["Annual_day"],
    pred["Predicted_Annual_day"],
    s=12,
    alpha=0.35
)

minimum = min(
    pred["Annual_day"].min(),
    pred["Predicted_Annual_day"].min()
)

maximum = max(
    pred["Annual_day"].max(),
    pred["Predicted_Annual_day"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=1.5
)

plt.xlabel(
    "Observed Annual Daytime SUHI"
)

plt.ylabel(
    "Predicted Annual Daytime SUHI"
)

plt.title(
    "Observed versus Predicted Annual Daytime SUHI"
)

plt.tight_layout()

plt.savefig(
    out / "Figure_4_Observed_vs_Predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 5. MORAN'S I SENSITIVITY
# =====================================================

moran = pd.read_csv(
    "YCEO_UHI_Moran_Sensitivity.csv"
)

plt.figure(figsize=(7, 5))

plt.plot(
    moran["k"],
    moran["Moran_I"],
    marker="o",
    linewidth=1.8
)

plt.xlabel(
    "Number of nearest neighbors (k)"
)

plt.ylabel(
    "Moran's I"
)

plt.title(
    "Residual Spatial Autocorrelation Sensitivity"
)

plt.xticks(
    moran["k"]
)

plt.tight_layout()

plt.savefig(
    out / "Figure_5_Moran_Sensitivity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =====================================================
# 6. FINAL RESULTS TABLE
# =====================================================

best_shap = (
    shap
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
)

top_features = best_shap.head(5)


final_table = pd.DataFrame({

    "Result": [
        "Number of original YCEO clusters",
        "Final modeling observations",
        "Number of predictors",
        "Spatial blocks",
        "Spatial CV folds",
        "Mean spatial R²",
        "SD of spatial R²",
        "Mean spatial RMSE",
        "SD of spatial RMSE",
        "Mean spatial MAE",
        "SD of spatial MAE",
        "Single holdout R²",
        "Single holdout RMSE",
        "Single holdout MAE",
        "Moran's I (k=8)",
        "Moran's I p-value",
        "Top SHAP predictor",
        "Second SHAP predictor",
        "Third SHAP predictor",
        "Fourth SHAP predictor",
        "Fifth SHAP predictor"
    ],

    "Value": [

        10136,
        9790,
        9,
        182,
        5,

        cv["R2"].mean(),
        cv["R2"].std(ddof=1),

        cv["RMSE"].mean(),
        cv["RMSE"].std(ddof=1),

        cv["MAE"].mean(),
        cv["MAE"].std(ddof=1),

        0.382648,
        0.614984,
        0.450956,

        moran.loc[
            moran["k"] == 8,
            "Moran_I"
        ].iloc[0],

        moran.loc[
            moran["k"] == 8,
            "Permutation_p"
        ].iloc[0],

        top_features.iloc[0]["Feature"],
        top_features.iloc[1]["Feature"],
        top_features.iloc[2]["Feature"],
        top_features.iloc[3]["Feature"],
        top_features.iloc[4]["Feature"]
    ]
})


final_table.to_csv(
    out / "Table_1_Final_Results.csv",
    index=False
)


# =====================================================
# 7. PRINT SUMMARY
# =====================================================

print()
print("======================================")
print("FINAL PAPER FIGURES")
print("======================================")

print()
print("Created:")

for file in sorted(out.iterdir()):
    print(
        file.name
    )

print()
print("======================================")
print("KEY RESULTS")
print("======================================")

print(
    "Mean spatial R²:",
    round(
        cv["R2"].mean(),
        6
    ),
    "±",
    round(
        cv["R2"].std(ddof=1),
        6
    )
)

print(
    "Mean spatial RMSE:",
    round(
        cv["RMSE"].mean(),
        6
    ),
    "±",
    round(
        cv["RMSE"].std(ddof=1),
        6
    )
)

print(
    "Mean spatial MAE:",
    round(
        cv["MAE"].mean(),
        6
    ),
    "±",
    round(
        cv["MAE"].std(ddof=1),
        6
    )
)

print()
print("Top SHAP predictors:")

for _, row in (
    best_shap
    .sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )
    .head(5)
    .iterrows()
):

    print(
        row["Feature"],
        ":",
        round(
            row["Mean_Absolute_SHAP"],
            6
        )
    )

print()
print(
    "All paper figures and table created."
)