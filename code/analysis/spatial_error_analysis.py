import pandas as pd
import numpy as np

# =====================================================
# LOAD FINAL PREDICTIONS
# =====================================================

df = pd.read_csv(
    "YCEO_UHI_Final_Predictions.csv"
)

print()
print("======================================")
print("SPATIAL ERROR ANALYSIS")
print("======================================")

print(
    "Total test observations:",
    len(df)
)


# =====================================================
# BASIC RESIDUAL STATISTICS
# =====================================================

df["Absolute_Error"] = (
    df["Residual"].abs()
)

df["Squared_Error"] = (
    df["Residual"] ** 2
)


print()
print("Overall residual statistics:")
print(
    df["Residual"].describe().to_string()
)


print()
print(
    "Mean absolute error:",
    round(
        df["Absolute_Error"].mean(),
        4
    )
)


# =====================================================
# LATITUDE BANDS
# =====================================================

def latitude_zone(lat):

    if lat < -60:
        return "Antarctic / High Southern"

    elif lat < -30:
        return "Southern Mid-Latitudes"

    elif lat < 0:
        return "Southern Tropics"

    elif lat < 30:
        return "Northern Tropics"

    elif lat < 60:
        return "Northern Mid-Latitudes"

    else:
        return "Arctic / High Northern"


df["Latitude_Zone"] = (
    df["Lat"]
    .apply(latitude_zone)
)


# =====================================================
# LATITUDE ERROR SUMMARY
# =====================================================

latitude_summary = (
    df.groupby(
        "Latitude_Zone"
    )
    .agg(
        Observations=("Code", "count"),
        Mean_Residual=("Residual", "mean"),
        MAE=("Absolute_Error", "mean"),
        RMSE=(
            "Squared_Error",
            lambda x: np.sqrt(x.mean())
        )
    )
    .reset_index()
)


print()
print("======================================")
print("ERROR BY LATITUDE ZONE")
print("======================================")

print(
    latitude_summary.to_string(
        index=False
    )
)


# =====================================================
# LONGITUDE REGIONS
# =====================================================

def longitude_zone(lon):

    if lon < -120:
        return "Pacific Americas"

    elif lon < -60:
        return "Americas"

    elif lon < 0:
        return "Atlantic / Western Eurasia"

    elif lon < 60:
        return "Africa / Europe / Western Asia"

    elif lon < 120:
        return "South / East Asia"

    else:
        return "Pacific / Oceania"


df["Longitude_Zone"] = (
    df["Lon"]
    .apply(longitude_zone)
)


# =====================================================
# LONGITUDE ERROR SUMMARY
# =====================================================

longitude_summary = (
    df.groupby(
        "Longitude_Zone"
    )
    .agg(
        Observations=("Code", "count"),
        Mean_Residual=("Residual", "mean"),
        MAE=("Absolute_Error", "mean"),
        RMSE=(
            "Squared_Error",
            lambda x: np.sqrt(x.mean())
        )
    )
    .reset_index()
)


print()
print("======================================")
print("ERROR BY LONGITUDE REGION")
print("======================================")

print(
    longitude_summary.to_string(
        index=False
    )
)


# =====================================================
# ERROR QUANTILES
# =====================================================

quantiles = (
    df["Absolute_Error"]
    .quantile(
        [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


print()
print("======================================")
print("ABSOLUTE ERROR QUANTILES")
print("======================================")

print(
    quantiles.to_string()
)


# =====================================================
# LARGEST ERRORS
# =====================================================

largest_errors = (
    df.sort_values(
        "Absolute_Error",
        ascending=False
    )
    .head(50)
)


largest_errors[
    [
        "Code",
        "Lat",
        "Lon",
        "Annual_day",
        "Predicted_Annual_day",
        "Residual",
        "Absolute_Error"
    ]
].to_csv(
    "YCEO_UHI_Largest_Prediction_Errors.csv",
    index=False
)


# =====================================================
# SAVE SUMMARIES
# =====================================================

latitude_summary.to_csv(
    "YCEO_UHI_Error_By_Latitude.csv",
    index=False
)

longitude_summary.to_csv(
    "YCEO_UHI_Error_By_Longitude.csv",
    index=False
)

quantiles.to_csv(
    "YCEO_UHI_Error_Quantiles.csv"
)


# =====================================================
# SAVE FULL ERROR DATA
# =====================================================

df.to_csv(
    "YCEO_UHI_Spatial_Error_Analysis.csv",
    index=False
)


# =====================================================
# COMPLETE
# =====================================================

print()
print("======================================")
print("FILES CREATED")
print("======================================")

print(
    "YCEO_UHI_Spatial_Error_Analysis.csv"
)

print(
    "YCEO_UHI_Error_By_Latitude.csv"
)

print(
    "YCEO_UHI_Error_By_Longitude.csv"
)

print(
    "YCEO_UHI_Error_Quantiles.csv"
)

print(
    "YCEO_UHI_Largest_Prediction_Errors.csv"
)

print()
print("Spatial error analysis complete.")