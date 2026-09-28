import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
N_FOLDS = 5

FEATURES = [
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

TARGET = "Annual_day"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print()
print("==============================================")
print("GEOGRAPHIC DISTRIBUTION-SHIFT BENCHMARK")
print("==============================================")

print()
print("Raw dataset:", df.shape)


# ============================================================
# CLEANING
# ============================================================

df = df.dropna(
    subset=FEATURES + [
        TARGET,
        "Lat",
        "Lon"
    ]
).copy()

df = df[
    df["Population"] >= 0
].copy()

df = df[
    (df["NDVI"] >= -1) &
    (df["NDVI"] <= 1)
].copy()


# ============================================================
# SPATIAL BLOCKS
# ============================================================

df["lat_block"] = np.floor(
    df["Lat"] / 10
).astype(int)

df["lon_block"] = np.floor(
    df["Lon"] / 10
).astype(int)

df["spatial_block"] = (
    df["lat_block"].astype(str)
    + "_"
    + df["lon_block"].astype(str)
)


X = df[FEATURES]
y = df[TARGET]
groups = df["spatial_block"]


print("Modeling observations:", len(df))
print("Spatial blocks:", groups.nunique())


# ============================================================
# PCA REPRESENTATION OF ENVIRONMENTAL / URBAN SPACE
# ============================================================
#
# We build a standardized predictor representation.
# This is NOT used to train the predictive model.
#
# It is used only to quantify how far a test observation
# lies from the predictor distribution of its training data.
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

pca = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

X_pca = pca.fit_transform(
    X_scaled
)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]


# ============================================================
# XGBOOST CONFIGURATION
# ============================================================

def create_model():

    return XGBRegressor(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=6,

        min_child_weight=5,

        subsample=0.8,

        colsample_bytree=0.9,

        reg_lambda=1.0,

        objective="reg:squarederror",

        eval_metric="rmse",

        random_state=RANDOM_STATE,

        n_jobs=-1,

        tree_method="hist"
    )


# ============================================================
# FIVE-FOLD SPATIAL VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)


fold_results = []
observation_results = []


for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print()
    print("----------------------------------------------")
    print("Fold", fold)
    print("----------------------------------------------")


    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]


    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    r2 = r2_score(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )


    print(
        "R²   =",
        round(r2, 6)
    )

    print(
        "RMSE =",
        round(rmse, 6)
    )

    print(
        "MAE  =",
        round(mae, 6)
    )


    # --------------------------------------------------------
    # TRAINING-DISTRIBUTION CENTROID
    # --------------------------------------------------------

    train_pc = df.iloc[
        train_idx
    ][
        ["PC1", "PC2"]
    ].to_numpy()

    test_pc = df.iloc[
        test_idx
    ][
        ["PC1", "PC2"]
    ].to_numpy()


    train_centroid = (
        train_pc.mean(
            axis=0
        )
    )


    # Euclidean distance from the center of the training
    # predictor distribution.
    shift_distance = np.sqrt(
        (
            test_pc
            - train_centroid
        ) ** 2
    ).sum(axis=1)


    # --------------------------------------------------------
    # ABSOLUTE ERROR
    # --------------------------------------------------------

    absolute_error = np.abs(
        y_test.to_numpy()
        - predictions
    )


    # --------------------------------------------------------
    # STORE OBSERVATION-LEVEL RESULTS
    # --------------------------------------------------------

    test_meta = df.iloc[
        test_idx
    ][
        [
            "Code",
            "Lat",
            "Lon",
            "spatial_block",
            "PC1",
            "PC2"
        ]
    ].copy()

    test_meta["Fold"] = fold

    test_meta["Observed"] = (
        y_test.to_numpy()
    )

    test_meta["Predicted"] = (
        predictions
    )

    test_meta["Absolute_Error"] = (
        absolute_error
    )

    test_meta["Shift_Distance"] = (
        shift_distance
    )

    observation_results.append(
        test_meta
    )


    # --------------------------------------------------------
    # FOLD-LEVEL RESULT
    # --------------------------------------------------------

    fold_results.append({

        "Fold": fold,

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae,

        "Mean_Shift_Distance":
            shift_distance.mean(),

        "Median_Shift_Distance":
            np.median(
                shift_distance
            ),

        "Mean_Absolute_Error":
            absolute_error.mean()
    })


# ============================================================
# COMBINE RESULTS
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# SHIFT-ERROR RELATIONSHIP
# ============================================================

correlation = np.corrcoef(
    obs_df["Shift_Distance"],
    obs_df["Absolute_Error"]
)[0, 1]


print()
print("==============================================")
print("DISTRIBUTION-SHIFT RESULTS")
print("==============================================")

print()
print(
    fold_df.to_string(
        index=False
    )
)


# ============================================================
# ERROR BY SHIFT QUANTILE
# ============================================================

obs_df["Shift_Quartile"] = pd.qcut(
    obs_df["Shift_Distance"],
    q=4,
    labels=[
        "Q1 - closest",
        "Q2",
        "Q3",
        "Q4 - farthest"
    ],
    duplicates="drop"
)


shift_summary = (
    obs_df
    .groupby(
        "Shift_Quartile",
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
        ),

        RMSE=(
            "Absolute_Error",
            lambda x: np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        )
    )
    .reset_index()
)


print()
print("==============================================")
print("ERROR BY DISTRIBUTION-SHIFT QUARTILE")
print("==============================================")

print(
    shift_summary.to_string(
        index=False
    )
)


print()
print(
    "Correlation between shift distance and absolute error:"
)

print(
    round(
        correlation,
        6
    )
)


# ============================================================
# OVERALL CROSS-VALIDATION SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "R2",
        "RMSE",
        "MAE"
    ],

    "Mean": [
        fold_df["R2"].mean(),
        fold_df["RMSE"].mean(),
        fold_df["MAE"].mean()
    ],

    "SD": [
        fold_df["R2"].std(
            ddof=1
        ),

        fold_df["RMSE"].std(
            ddof=1
        ),

        fold_df["MAE"].std(
            ddof=1
        )
    ]
})


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_DistributionShift_FoldResults.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_DistributionShift_ObservationResults.csv",
    index=False
)

shift_summary.to_csv(
    "YCEO_DistributionShift_ErrorByShift.csv",
    index=False
)

summary.to_csv(
    "YCEO_DistributionShift_Summary.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==============================================")
print("OVERALL FIVE-FOLD RESULTS")
print("==============================================")

print(
    summary.to_string(
        index=False
    )
)

print()
print("Files created:")

print(
    "YCEO_DistributionShift_FoldResults.csv"
)

print(
    "YCEO_DistributionShift_ObservationResults.csv"
)

print(
    "YCEO_DistributionShift_ErrorByShift.csv"
)

print(
    "YCEO_DistributionShift_Summary.csv"
)

print()
print(
    "Distribution-shift benchmark complete."
)