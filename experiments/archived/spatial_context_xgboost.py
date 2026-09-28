import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
N_FOLDS = 5

K_NEIGHBORS = 8

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


X = df[FEATURES].copy()
y = df[TARGET].copy()
groups = df["spatial_block"].copy()


print()
print("============================================================")
print("SPATIAL-CONTEXT XGBOOST")
print("============================================================")

print("Dataset:", df.shape)
print("Predictors:", len(FEATURES))
print("Spatial blocks:", groups.nunique())
print("Neighbors:", K_NEIGHBORS)


# ============================================================
# XGBOOST
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
# GEOGRAPHIC DISTANCE
# ============================================================

def haversine_distance_matrix(
    lat,
    lon
):

    lat_rad = np.radians(
        lat
    )

    lon_rad = np.radians(
        lon
    )

    dlat = (
        lat_rad[:, None]
        - lat_rad[None, :]
    )

    dlon = (
        lon_rad[:, None]
        - lon_rad[None, :]
    )

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat_rad[:, None])
        *
        np.cos(lat_rad[None, :])
        *
        np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arcsin(
        np.sqrt(
            np.clip(
                a,
                0.0,
                1.0
            )
        )
    )

    return 6371.0 * c


# ============================================================
# NEIGHBOR FEATURES
# ============================================================
#
# For each target observation:
#
#   1. Find geographic neighbors.
#   2. Calculate predictor summaries of those neighbors.
#
# ONLY predictor values are used.
#
# For training observations, the observation itself is excluded.
#
# For outer test observations, only OUTER TRAINING observations
# are used as neighbors.
# ============================================================

def build_train_neighbor_features(
    X_train,
    lat_train,
    lon_train
):

    n_train = len(
        X_train
    )

    distances = haversine_distance_matrix(
        np.asarray(lat_train),
        np.asarray(lon_train)
    )

    np.fill_diagonal(
        distances,
        np.inf
    )

    records = []


    for i in range(
        n_train
    ):

        k = min(
            K_NEIGHBORS,
            n_train - 1
        )

        nearest_idx = np.argpartition(
            distances[i],
            k
        )[:k]


        neighbor_values = (
            X_train.iloc[
                nearest_idx
            ]
        )


        row = {}


        for feature in FEATURES:

            values = neighbor_values[
                feature
            ].to_numpy()


            row[
                f"NeighborMean_{feature}"
            ] = values.mean()

            row[
                f"NeighborStd_{feature}"
            ] = values.std(
                ddof=0
            )


        row[
            "NeighborMean_Distance_km"
        ] = distances[
            i,
            nearest_idx
        ].mean()


        records.append(
            row
        )


    return pd.DataFrame(
        records,
        index=X_train.index
    )


def build_test_neighbor_features(
    X_train,
    lat_train,
    lon_train,
    X_test,
    lat_test,
    lon_test
):

    train_lat = np.asarray(
        lat_train
    )

    train_lon = np.asarray(
        lon_train
    )

    test_lat = np.asarray(
        lat_test
    )

    test_lon = np.asarray(
        lon_test
    )


    n_test = len(
        X_test
    )

    n_train = len(
        X_train
    )


    records = []


    # --------------------------------------------------------
    # Calculate geographic distance for each test observation
    # to every OUTER TRAINING observation.
    # --------------------------------------------------------

    lat1 = np.radians(
        test_lat
    )[:, None]

    lon1 = np.radians(
        test_lon
    )[:, None]

    lat2 = np.radians(
        train_lat
    )[None, :]

    lon2 = np.radians(
        train_lon
    )[None, :]


    dlat = lat1 - lat2

    dlon = lon1 - lon2


    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        *
        np.cos(lat2)
        *
        np.sin(dlon / 2.0) ** 2
    )


    c = 2.0 * np.arcsin(
        np.sqrt(
            np.clip(
                a,
                0.0,
                1.0
            )
        )
    )


    distances = (
        6371.0 * c
    )


    for i in range(
        n_test
    ):

        k = min(
            K_NEIGHBORS,
            n_train
        )


        nearest_idx = np.argpartition(
            distances[i],
            k
        )[:k]


        neighbor_values = (
            X_train.iloc[
                nearest_idx
            ]
        )


        row = {}


        for feature in FEATURES:

            values = neighbor_values[
                feature
            ].to_numpy()


            row[
                f"NeighborMean_{feature}"
            ] = values.mean()

            row[
                f"NeighborStd_{feature}"
            ] = values.std(
                ddof=0
            )


        row[
            "NeighborMean_Distance_km"
        ] = distances[
            i,
            nearest_idx
        ].mean()


        records.append(
            row
        )


    return pd.DataFrame(
        records,
        index=X_test.index
    )


# ============================================================
# OUTER SPATIAL CV
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)


baseline_results = []
spatial_results = []

observation_results = []


for fold, (
    train_idx,
    test_idx
) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    print()
    print("============================================================")
    print(
        "OUTER SPATIAL FOLD",
        fold
    )
    print("============================================================")


    X_train = X.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_train = y.iloc[
        train_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    lat_train = df.iloc[
        train_idx
    ]["Lat"].to_numpy()

    lon_train = df.iloc[
        train_idx
    ]["Lon"].to_numpy()

    lat_test = df.iloc[
        test_idx
    ]["Lat"].to_numpy()

    lon_test = df.iloc[
        test_idx
    ]["Lon"].to_numpy()


    # ========================================================
    # BASELINE XGBOOST
    # ========================================================

    baseline = make_model()

    baseline.fit(
        X_train,
        y_train
    )

    baseline_pred = baseline.predict(
        X_test
    )


    baseline_r2 = r2_score(
        y_test,
        baseline_pred
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_pred
        )
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_pred
    )


    # ========================================================
    # TRAINING-SET NEIGHBOR FEATURES
    # ========================================================

    train_context = build_train_neighbor_features(
        X_train,
        lat_train,
        lon_train
    )


    test_context = build_test_neighbor_features(
        X_train,
        lat_train,
        lon_train,
        X_test,
        lat_test,
        lon_test
    )


    # Make sure indices align.
    train_context = train_context.set_axis(
        X_train.index
    )

    test_context = test_context.set_axis(
        X_test.index
    )


    # ========================================================
    # CONCATENATE ORIGINAL + SPATIAL CONTEXT FEATURES
    # ========================================================

    X_train_context = pd.concat(
        [
            X_train,
            train_context
        ],
        axis=1
    )

    X_test_context = pd.concat(
        [
            X_test,
            test_context
        ],
        axis=1
    )


    # ========================================================
    # SPATIAL-CONTEXT XGBOOST
    # ========================================================

    spatial_model = make_model()

    spatial_model.fit(
        X_train_context,
        y_train
    )

    spatial_pred = spatial_model.predict(
        X_test_context
    )


    spatial_r2 = r2_score(
        y_test,
        spatial_pred
    )

    spatial_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            spatial_pred
        )
    )

    spatial_mae = mean_absolute_error(
        y_test,
        spatial_pred
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("BASELINE XGBOOST")

    print(
        "R²   =",
        round(
            baseline_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            baseline_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            baseline_mae,
            6
        )
    )


    print()
    print("SPATIAL-CONTEXT XGBOOST")

    print(
        "R²   =",
        round(
            spatial_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            spatial_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            spatial_mae,
            6
        )
    )


    # ========================================================
    # STORE
    # ========================================================

    baseline_results.append({

        "Fold": fold,

        "Model": "Baseline_XGBoost",

        "R2": baseline_r2,

        "RMSE": baseline_rmse,

        "MAE": baseline_mae
    })


    spatial_results.append({

        "Fold": fold,

        "Model": "Spatial_Context_XGBoost",

        "R2": spatial_r2,

        "RMSE": spatial_rmse,

        "MAE": spatial_mae
    })


    # ========================================================
    # OBSERVATION RESULTS
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

    meta["Baseline_Predicted"] = (
        baseline_pred
    )

    meta["Baseline_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - baseline_pred
        )
    )

    meta["SpatialContext_Predicted"] = (
        spatial_pred
    )

    meta["SpatialContext_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - spatial_pred
        )
    )

    meta[
        "Mean_Neighbor_Distance_km"
    ] = test_context[
        "NeighborMean_Distance_km"
    ].to_numpy()


    observation_results.append(
        meta
    )


# ============================================================
# COMBINE
# ============================================================

baseline_df = pd.DataFrame(
    baseline_results
)

spatial_df = pd.DataFrame(
    spatial_results
)

comparison_df = pd.concat(
    [
        baseline_df,
        spatial_df
    ],
    ignore_index=True
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    comparison_df
    .groupby(
        "Model"
    )
    .agg(

        Mean_R2=(
            "R2",
            "mean"
        ),

        SD_R2=(
            "R2",
            "std"
        ),

        Mean_RMSE=(
            "RMSE",
            "mean"
        ),

        SD_RMSE=(
            "RMSE",
            "std"
        ),

        Mean_MAE=(
            "MAE",
            "mean"
        ),

        SD_MAE=(
            "MAE",
            "std"
        )
    )
    .reset_index()
)


# ============================================================
# PAIRED DIFFERENCES
# ============================================================

paired_rows = []


for fold in sorted(
    comparison_df["Fold"].unique()
):

    base = baseline_df[
        baseline_df["Fold"] == fold
    ].iloc[0]

    spatial = spatial_df[
        spatial_df["Fold"] == fold
    ].iloc[0]


    paired_rows.append({

        "Fold": fold,

        "Delta_R2":
            spatial["R2"]
            - base["R2"],

        "Delta_RMSE":
            spatial["RMSE"]
            - base["RMSE"],

        "Delta_MAE":
            spatial["MAE"]
            - base["MAE"]
    })


paired_df = pd.DataFrame(
    paired_rows
)


# ============================================================
# PRINT FINAL
# ============================================================

print()
print("============================================================")
print("FINAL MODEL SUMMARY")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("PAIRED FOLD DIFFERENCES")
print("============================================================")

print(
    paired_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

comparison_df.to_csv(
    "YCEO_SpatialContext_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_SpatialContext_Model_Summary.csv",
    index=False
)

paired_df.to_csv(
    "YCEO_SpatialContext_Paired_Differences.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_SpatialContext_Observation_Results.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_SpatialContext_Model_Comparison.csv"
)

print(
    "YCEO_SpatialContext_Model_Summary.csv"
)

print(
    "YCEO_SpatialContext_Paired_Differences.csv"
)

print(
    "YCEO_SpatialContext_Observation_Results.csv"
)

print()
print(
    "Spatial-context XGBoost experiment complete."
)