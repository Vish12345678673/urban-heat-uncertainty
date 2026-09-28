import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
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

# Number of robust reweighting rounds.
ROBUST_ROUNDS = 3

# Maximum group-weight multiplier.
MAX_GROUP_WEIGHT = 3.0

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

df = pd.read_csv(
    DATA_FILE
)

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
# GROUP ROBUST WEIGHTS
# ============================================================

def train_group_robust_model(
    X_train,
    y_train,
    train_groups
):

    # Start with equal observation weights.
    weights = np.ones(
        len(X_train),
        dtype=float
    )

    final_model = None

    for robust_round in range(
        ROBUST_ROUNDS
    ):

        model = make_model()

        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )

        predictions = model.predict(
            X_train
        )

        squared_error = (
            y_train.to_numpy()
            - predictions
        ) ** 2


        temp = pd.DataFrame({

            "Group":
                np.asarray(train_groups),

            "Error":
                squared_error
        })


        group_loss = (
            temp
            .groupby("Group")["Error"]
            .mean()
        )


        # ----------------------------------------------------
        # Normalize group losses.
        # ----------------------------------------------------

        mean_loss = group_loss.mean()

        if mean_loss <= 1e-12:
            final_model = model
            break

        group_multiplier = (
            group_loss
            / mean_loss
        )

        # Conservative clipping prevents one tiny group
        # from dominating the optimization.
        group_multiplier = group_multiplier.clip(
            lower=0.5,
            upper=MAX_GROUP_WEIGHT
        )


        # ----------------------------------------------------
        # Assign group-level weights to observations.
        # ----------------------------------------------------

        new_weights = np.array([
            group_multiplier[g]
            for g in train_groups
        ])


        # Normalize mean observation weight to 1.
        new_weights = (
            new_weights
            / new_weights.mean()
        )


        weights = new_weights

        final_model = model


    # Retrain one final model using the final robust weights.
    final_model = make_model()

    final_model.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    return final_model, weights


# ============================================================
# CV
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)

baseline_results = []
robust_results = []

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
    print("================================================")
    print("SPATIAL FOLD", fold)
    print("================================================")


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

    train_groups = groups.iloc[
        train_idx
    ].to_numpy()


    # ========================================================
    # BASELINE
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
    # GROUP-ROBUST MODEL
    # ========================================================

    robust_model, final_weights = (
        train_group_robust_model(
            X_train,
            y_train,
            train_groups
        )
    )

    robust_pred = robust_model.predict(
        X_test
    )


    robust_r2 = r2_score(
        y_test,
        robust_pred
    )

    robust_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            robust_pred
        )
    )

    robust_mae = mean_absolute_error(
        y_test,
        robust_pred
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
    print("GROUP-ROBUST XGBOOST")

    print(
        "R²   =",
        round(
            robust_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            robust_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            robust_mae,
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


    robust_results.append({

        "Fold": fold,

        "Model": "Group_Robust_XGBoost",

        "R2": robust_r2,

        "RMSE": robust_rmse,

        "MAE": robust_mae
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

    meta["Robust_Predicted"] = (
        robust_pred
    )

    meta["Robust_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - robust_pred
        )
    )

    observation_results.append(
        meta
    )


# ============================================================
# COMBINE
# ============================================================

baseline_df = pd.DataFrame(
    baseline_results
)

robust_df = pd.DataFrame(
    robust_results
)

comparison_df = pd.concat(
    [
        baseline_df,
        robust_df
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
    .groupby("Model")
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
# GLOBAL ERROR QUARTILES
# ============================================================

obs_df["Error_Quartile"] = pd.qcut(
    obs_df["Baseline_Absolute_Error"],
    q=4,
    labels=[
        "Q1 - easiest",
        "Q2",
        "Q3",
        "Q4 - hardest"
    ],
    duplicates="drop"
)


hardness_summary = (
    obs_df
    .groupby(
        "Error_Quartile",
        observed=True
    )
    .agg(

        Observations=(
            "Code",
            "count"
        ),

        Baseline_MAE=(
            "Baseline_Absolute_Error",
            "mean"
        ),

        Robust_MAE=(
            "Robust_Absolute_Error",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("================================================")
print("MODEL SUMMARY")
print("================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("================================================")
print("ERROR-QUARTILE PERFORMANCE")
print("================================================")

print(
    hardness_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

comparison_df.to_csv(
    "YCEO_GroupRobust_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_GroupRobust_Model_Summary.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_GroupRobust_Observation_Results.csv",
    index=False
)

hardness_summary.to_csv(
    "YCEO_GroupRobust_Error_Quartiles.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("================================================")
print("FILES CREATED")
print("================================================")

print(
    "YCEO_GroupRobust_Model_Comparison.csv"
)

print(
    "YCEO_GroupRobust_Model_Summary.csv"
)

print(
    "YCEO_GroupRobust_Observation_Results.csv"
)

print(
    "YCEO_GroupRobust_Error_Quartiles.csv"
)

print()
print(
    "Group-robust XGBoost experiment complete."
)