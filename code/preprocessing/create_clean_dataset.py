import pandas as pd
import numpy as np
import os

print("Loading 21 YCEO enhanced batches...")

files = [
    f"YCEO_Enhanced_Batch_{i}.csv"
    for i in range(21)
]

dataframes = []

for file in files:

    if not os.path.exists(file):
        print("ERROR: Missing:", file)
        raise SystemExit

    print("Reading:", file)

    df = pd.read_csv(file)

    dataframes.append(df)


print("\nMerging files...")

df = pd.concat(
    dataframes,
    ignore_index=True
)


print("Raw rows:", len(df))
print("Raw columns:", len(df.columns))
print("Unique clusters:", df["Code"].nunique())


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

target = "Annual_day"


# =====================================================
# CLEANING
# =====================================================

print("\nCleaning dataset...")

# Target must exist
df = df.dropna(
    subset=[target]
).copy()

print(
    "After removing missing Annual_day:",
    len(df)
)


# Remove impossible population
df = df[
    df["Population"] >= 0
].copy()

print(
    "After removing negative population:",
    len(df)
)


# Valid NDVI range
df = df[
    (df["NDVI"] >= -1) &
    (df["NDVI"] <= 1)
].copy()


# Remove rows missing required predictors
df = df.dropna(
    subset=features
).copy()


# Flag negative NDVI
df["NDVI_negative_flag"] = (
    df["NDVI"] < 0
).astype(int)


# =====================================================
# FINAL CHECK
# =====================================================

print("\n====================================")
print("FINAL DATASET")
print("====================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique clusters:", df["Code"].nunique())

print("\nMissing predictor values:")

print(
    df[features]
    .isna()
    .sum()
    .to_string()
)

print(
    "\nNegative population:",
    (df["Population"] < 0).sum()
)

print(
    "Negative elevation:",
    (df["Elevation"] < 0).sum()
)

print(
    "Negative NDVI:",
    (df["NDVI"] < 0).sum()
)


# =====================================================
# SAVE
# =====================================================

output = (
    "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"
)

df.to_csv(
    output,
    index=False
)

print(
    "\nSaved:",
    output
)

print("\nDONE.")