import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


# =====================================================
# LOAD DATA
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
    "Mean_Monthly_Precip_mm",
    "Mean_Temp_C",
    "Builtup"
]


# =====================================================
# MATCH ACTUAL VALUES
# =====================================================

actual = original[
    ["Code"] + features
].copy()


df = shap_df[
    ["Code"] + features
].merge(
    actual,
    on="Code",
    suffixes=("_shap", "_actual")
)


# =====================================================
# CREATE FIGURES
# =====================================================

for feature in features:

    x = df[
        feature + "_actual"
    ].to_numpy()

    y = df[
        feature + "_shap"
    ].to_numpy()


    # Remove invalid values
    valid = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]


    # Sort by feature value
    order = np.argsort(x)

    x_sorted = x[order]
    y_sorted = y[order]


    # -------------------------------------------------
    # Smooth curve
    # -------------------------------------------------

    window = min(
        101,
        len(y_sorted) - 1
    )

    if window % 2 == 0:
        window -= 1

    if window >= 7:

        smooth = savgol_filter(
            y_sorted,
            window_length=window,
            polyorder=2
        )

    else:

        smooth = y_sorted


    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        x,
        y,
        s=10,
        alpha=0.25
    )

    plt.plot(
        x_sorted,
        smooth,
        linewidth=2
    )

    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel(
        feature
    )

    plt.ylabel(
        "SHAP value"
    )

    plt.title(
        "SHAP Dependence: " + feature
    )

    plt.tight_layout()


    filename = (
        "Final_SHAP_Dependence_"
        + feature
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "Created:",
        filename
    )


print()
print(
    "Final SHAP dependence plots complete."
)