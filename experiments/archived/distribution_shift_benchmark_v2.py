import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
N_FOLDS = 5
K_NEIGHBORS = 5

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
# LOAD
# ============================================================

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=FEATURES + ["Lat", "Lon", TARGET]
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


# ============================================================
# MODEL
# ============================================================

def make_model():

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
# 5-FOLD SPATIAL CV
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
    print("======================================")
    print("SPATIAL FOLD", fold)
    print("======================================")


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


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    model = make_model()

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )


    # ========================================================
    # MODEL METRICS
    # ========================================================

    r2 = r2_score(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )


    print(
        "R²   =", round(r2, 6)
    )

    print(
        "RMSE =", round(rmse, 6)
    )

    print(
        "MAE  =", round(mae, 6)
    )


    # ========================================================
    # LEAKAGE-FREE DISTRIBUTION DISTANCE
    # ========================================================
    #
    # IMPORTANT:
    # The scaler is fitted ONLY on training observations.
    #
    # The nearest-neighbor model is also fitted ONLY on
    # training observations.
    #
    # Therefore the test distribution distance is independent
    # of the test feature distribution transformation.
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    nn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS,
        metric="euclidean",
        n_jobs=-1
    )

    nn.fit(
        X_train_scaled
    )


    distances, _ = nn.kneighbors(
        X_test_scaled
    )


    # Mean distance to the k nearest training observations
    shift_distance = distances.mean(
        axis=1
    )


    # ========================================================
    # ABSOLUTE ERROR
    # ========================================================

    absolute_error = np.abs(
        y_test.to_numpy()
        - prediction
    )


    # ========================================================
    # STORE OBSERVATION RESULTS
    # ========================================================

    meta = df.iloc[
        test_idx
    ][
        [
            "Code",
            "Lat",
            "Lon",
            "spatial_block"
        ]
    ].copy()

    meta["Fold"] = fold

    meta["Observed"] = (
        y_test.to_numpy()
    )

    meta["Predicted"] = (
        prediction
    )

    meta["Absolute_Error"] = (
        absolute_error
    )

    meta["Shift_Distance"] = (
        shift_distance
    )

    observation_results.append(
        meta
    )


    # ========================================================
    # FOLD RESULT
    # ========================================================

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
# COMBINE
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# CORRELATION
# ============================================================

spearman = obs_df[
    [
        "Shift_Distance",
        "Absolute_Error"
    ]
].corr(
    method="spearman"
).iloc[0, 1]

pearson = obs_df[
    [
        "Shift_Distance",
        "Absolute_Error"
    ]
].corr(
    method="pearson"
).iloc[0, 1]


# ============================================================
# SHIFT QUARTILES
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
            lambda x:
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        )
    )
    .reset_index()
)


# ============================================================
# OVERALL CV
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
        fold_df["R2"].std(ddof=1),
        fold_df["RMSE"].std(ddof=1),
        fold_df["MAE"].std(ddof=1)
    ]
})


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_DistributionShiftV2_FoldResults.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_DistributionShiftV2_ObservationResults.csv",
    index=False
)

shift_summary.to_csv(
    "YCEO_DistributionShiftV2_ErrorByShift.csv",
    index=False
)

summary.to_csv(
    "YCEO_DistributionShiftV2_Summary.csv",
    index=False
)


# ============================================================
# PRINT
# ============================================================

print()
print("======================================")
print("DISTRIBUTION-SHIFT V2")
print("======================================")

print()
print(
    fold_df.to_string(
        index=False
    )
)

print()
print("======================================")
print("ERROR BY SHIFT QUARTILE")
print("======================================")

print(
    shift_summary.to_string(
        index=False
    )
)

print()
print("======================================")
print("SHIFT / ERROR ASSOCIATION")
print("======================================")

print(
    "Pearson r :",
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
print("======================================")
print("OVERALL CV")
print("======================================")

print(
    summary.to_string(
        index=False
    )
)

print()
print("Files created:")

print(
    "YCEO_DistributionShiftV2_FoldResults.csv"
)

print(
    "YCEO_DistributionShiftV2_ObservationResults.csv"
)

print(
    "YCEO_DistributionShiftV2_ErrorByShift.csv"
)

print(
    "YCEO_DistributionShiftV2_Summary.csv"
)

print()
print(
    "Leakage-free distribution-shift benchmark complete."
)