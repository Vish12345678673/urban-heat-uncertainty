import numpy as np
import pandas as pd

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

N_ENVIRONMENTS = 5

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
# GEOGRAPHIC ENVIRONMENTS
# ============================================================
#
# Five longitude sectors.
#
# The boundaries are deterministic and use no target values.
# ============================================================

lon = df["Lon"].to_numpy()

environment = np.floor(
    (lon + 180.0)
    /
    72.0
).astype(int)

environment = np.clip(
    environment,
    0,
    N_ENVIRONMENTS - 1
)

df["Environment"] = environment


X = df[FEATURES]
y = df[TARGET]


print()
print("============================================================")
print("LEAVE-ONE-GEOGRAPHIC-ENVIRONMENT-OUT BENCHMARK")
print("============================================================")

print(
    "Dataset:",
    df.shape
)

print(
    "Features:",
    len(FEATURES)
)

print(
    "Environments:",
    N_ENVIRONMENTS
)


print()
print("Environment sizes:")

print(
    df["Environment"]
    .value_counts()
    .sort_index()
    .to_string()
)


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
# STORAGE
# ============================================================

erm_results = []

balanced_results = []

observation_results = []


# ============================================================
# LEAVE-ONE-ENVIRONMENT-OUT
# ============================================================

for held_out_env in range(
    N_ENVIRONMENTS
):

    print()
    print("============================================================")
    print(
        "HELD-OUT ENVIRONMENT",
        held_out_env
    )
    print("============================================================")


    test_mask = (
        df["Environment"]
        == held_out_env
    )

    train_mask = ~test_mask


    X_train = X.loc[
        train_mask
    ].copy()

    y_train = y.loc[
        train_mask
    ].copy()

    X_test = X.loc[
        test_mask
    ].copy()

    y_test = y.loc[
        test_mask
    ].copy()


    train_env = df.loc[
        train_mask,
        "Environment"
    ]


    print(
        "Training observations:",
        len(X_train)
    )

    print(
        "Testing observations:",
        len(X_test)
    )


    if len(X_test) == 0:

        continue


    # ========================================================
    # STANDARD ERM XGBOOST
    # ========================================================

    erm = make_model()

    erm.fit(
        X_train,
        y_train
    )

    erm_pred = erm.predict(
        X_test
    )


    erm_r2 = r2_score(
        y_test,
        erm_pred
    )

    erm_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            erm_pred
        )
    )

    erm_mae = mean_absolute_error(
        y_test,
        erm_pred
    )


    # ========================================================
    # ENVIRONMENT-BALANCED XGBOOST
    # ========================================================
    #
    # Each training environment receives equal total weight.
    #
    # If an environment has n observations:
    #
    #     observation weight ∝ 1 / n
    #
    # This prevents large geographic sectors from dominating
    # the training objective.
    # ========================================================

    environment_counts = (
        train_env
        .value_counts()
    )


    balanced_weights = np.array([
        1.0
        /
        environment_counts[
            env
        ]
        for env in train_env
    ])


    # Normalize mean weight to 1.
    balanced_weights = (
        balanced_weights
        /
        balanced_weights.mean()
    )


    balanced = make_model()

    balanced.fit(
        X_train,
        y_train,
        sample_weight=balanced_weights
    )

    balanced_pred = balanced.predict(
        X_test
    )


    balanced_r2 = r2_score(
        y_test,
        balanced_pred
    )

    balanced_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            balanced_pred
        )
    )

    balanced_mae = mean_absolute_error(
        y_test,
        balanced_pred
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("STANDARD ERM XGBOOST")

    print(
        "R²   =",
        round(
            erm_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            erm_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            erm_mae,
            6
        )
    )


    print()
    print("ENVIRONMENT-BALANCED XGBOOST")

    print(
        "R²   =",
        round(
            balanced_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            balanced_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            balanced_mae,
            6
        )
    )


    print()
    print("Changes:")

    print(
        "ΔR²   =",
        round(
            balanced_r2
            - erm_r2,
            6
        )
    )

    print(
        "ΔRMSE =",
        round(
            balanced_rmse
            - erm_rmse,
            6
        )
    )

    print(
        "ΔMAE  =",
        round(
            balanced_mae
            - erm_mae,
            6
        )
    )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    erm_results.append({

        "HeldOutEnvironment":
            held_out_env,

        "TestObservations":
            len(X_test),

        "R2":
            erm_r2,

        "RMSE":
            erm_rmse,

        "MAE":
            erm_mae
    })


    balanced_results.append({

        "HeldOutEnvironment":
            held_out_env,

        "TestObservations":
            len(X_test),

        "R2":
            balanced_r2,

        "RMSE":
            balanced_rmse,

        "MAE":
            balanced_mae
    })


    # ========================================================
    # OBSERVATION RESULTS
    # ========================================================

    meta = df.loc[
        test_mask,
        [
            "Code",
            "Lat",
            "Lon",
            "Environment"
        ]
    ].copy()


    meta["Observed"] = (
        y_test.to_numpy()
    )


    meta["ERM_Predicted"] = (
        erm_pred
    )

    meta["ERM_Absolute_Error"] = np.abs(
        y_test.to_numpy()
        -
        erm_pred
    )


    meta["Balanced_Predicted"] = (
        balanced_pred
    )

    meta["Balanced_Absolute_Error"] = np.abs(
        y_test.to_numpy()
        -
        balanced_pred
    )


    observation_results.append(
        meta
    )


# ============================================================
# DATAFRAMES
# ============================================================

erm_df = pd.DataFrame(
    erm_results
)

balanced_df = pd.DataFrame(
    balanced_results
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "R2",
        "RMSE",
        "MAE"
    ],

    "ERM_Mean": [

        erm_df["R2"].mean(),

        erm_df["RMSE"].mean(),

        erm_df["MAE"].mean()
    ],

    "Balanced_Mean": [

        balanced_df["R2"].mean(),

        balanced_df["RMSE"].mean(),

        balanced_df["MAE"].mean()
    ],

    "ERM_SD": [

        erm_df["R2"].std(
            ddof=1
        ),

        erm_df["RMSE"].std(
            ddof=1
        ),

        erm_df["MAE"].std(
            ddof=1
        )
    ],

    "Balanced_SD": [

        balanced_df["R2"].std(
            ddof=1
        ),

        balanced_df["RMSE"].std(
            ddof=1
        ),

        balanced_df["MAE"].std(
            ddof=1
        )
    ]
})


# ============================================================
# PAIRED DIFFERENCES
# ============================================================

paired = erm_df.merge(
    balanced_df,
    on=[
        "HeldOutEnvironment",
        "TestObservations"
    ],
    suffixes=(
        "_ERM",
        "_Balanced"
    )
)


paired["Delta_R2"] = (
    paired["R2_Balanced"]
    -
    paired["R2_ERM"]
)

paired["Delta_RMSE"] = (
    paired["RMSE_Balanced"]
    -
    paired["RMSE_ERM"]
)

paired["Delta_MAE"] = (
    paired["MAE_Balanced"]
    -
    paired["MAE_ERM"]
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("============================================================")
print("LEAVE-ONE-ENVIRONMENT-OUT SUMMARY")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("PAIRED ENVIRONMENT DIFFERENCES")
print("============================================================")

print(
    paired[
        [
            "HeldOutEnvironment",
            "TestObservations",
            "Delta_R2",
            "Delta_RMSE",
            "Delta_MAE"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# WORST HELD-OUT ENVIRONMENTS
# ============================================================

worst_erm = (
    erm_df
    .sort_values(
        "R2"
    )
    .head(3)
)

best_erm = (
    erm_df
    .sort_values(
        "R2",
        ascending=False
    )
    .head(3)
)


print()
print("============================================================")
print("STANDARD XGBOOST: WORST ENVIRONMENTS")
print("============================================================")

print(
    worst_erm.to_string(
        index=False
    )
)


print()
print("============================================================")
print("STANDARD XGBOOST: BEST ENVIRONMENTS")
print("============================================================")

print(
    best_erm.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

erm_df.to_csv(
    "YCEO_LeaveOneEnvironment_ERM.csv",
    index=False
)

balanced_df.to_csv(
    "YCEO_LeaveOneEnvironment_Balanced.csv",
    index=False
)

summary.to_csv(
    "YCEO_LeaveOneEnvironment_Summary.csv",
    index=False
)

paired.to_csv(
    "YCEO_LeaveOneEnvironment_Paired.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_LeaveOneEnvironment_Observations.csv",
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
    "YCEO_LeaveOneEnvironment_ERM.csv"
)

print(
    "YCEO_LeaveOneEnvironment_Balanced.csv"
)

print(
    "YCEO_LeaveOneEnvironment_Summary.csv"
)

print(
    "YCEO_LeaveOneEnvironment_Paired.csv"
)

print(
    "YCEO_LeaveOneEnvironment_Observations.csv"
)

print()
print(
    "Leave-one-geographic-environment benchmark complete."
)