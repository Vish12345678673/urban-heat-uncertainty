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
# RESULTS
# ============================================================

fold_results = []
regime_results = []


# ============================================================
# FIVE-FOLD SPATIAL VALIDATION
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)


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
        "R²   =",
        round(
            r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            mae,
            6
        )
    )


    fold_results.append({

        "Fold": fold,

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae
    })


    # ========================================================
    # CREATE REGIMES USING TRAINING DISTRIBUTION ONLY
    # ========================================================
    #
    # Each variable is divided into quartiles using training
    # quantiles. The same boundaries are then applied to test.
    #
    # This avoids using test values to define the regime rules.
    # ========================================================

    regime_variables = [
        "Mean_Temp_C",
        "Mean_Monthly_Precip_mm",
        "NDVI",
        "Builtup"
    ]


    train_regimes = {}
    test_regimes = {}


    for variable in regime_variables:

        q1, q2, q3 = (
            X_train[variable]
            .quantile(
                [0.25, 0.50, 0.75]
            )
            .values
        )


        def assign_regime(value):

            if value <= q1:
                return "Low"

            elif value <= q2:
                return "Medium-Low"

            elif value <= q3:
                return "Medium-High"

            else:
                return "High"


        train_regimes[
            variable
        ] = X_train[
            variable
        ].apply(
            assign_regime
        ).reset_index(
            drop=True
        )

        test_regimes[
            variable
        ] = X_test[
            variable
        ].apply(
            assign_regime
        ).reset_index(
            drop=True
        )


    # ========================================================
    # COMBINE REGIMES
    # ========================================================

    train_regime_df = pd.DataFrame(
        train_regimes
    )

    test_regime_df = pd.DataFrame(
        test_regimes
    )


    train_regime_code = (
        train_regime_df[
            regime_variables
        ]
        .astype(str)
        .agg(
            "|".join,
            axis=1
        )
    )

    test_regime_code = (
        test_regime_df[
            regime_variables
        ]
        .astype(str)
        .agg(
            "|".join,
            axis=1
        )
    )


    # ========================================================
    # TRAINING REGIME FREQUENCY
    # ========================================================

    train_counts = (
        train_regime_code
        .value_counts()
        .to_dict()
    )


    # ========================================================
    # TEST REGIME ANALYSIS
    # ========================================================

    test_absolute_error = (
        np.abs(
            y_test.to_numpy()
            - prediction
        )
    )


    test_regimes_out = (
        test_regime_code
        .to_numpy()
    )


    for regime in np.unique(
        test_regimes_out
    ):

        mask = (
            test_regimes_out
            == regime
        )


        n_test = int(
            mask.sum()
        )


        error = (
            test_absolute_error[
                mask
            ]
        )


        training_count = (
            train_counts
            .get(
                regime,
                0
            )
        )


        regime_results.append({

            "Fold": fold,

            "Regime": regime,

            "Test_Observations":
                n_test,

            "Training_Regime_Count":
                training_count,

            "Training_Regime_Frequency":
                training_count
                / len(train_regime_code),

            "Mean_Absolute_Error":
                error.mean(),

            "RMSE":
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                ),

            "Mean_Error":
                (
                    y_test.to_numpy()[mask]
                    - prediction[mask]
                ).mean()
        })


# ============================================================
# DATAFRAMES
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

regime_df = pd.DataFrame(
    regime_results
)


# ============================================================
# RARE VS COMMON REGIMES
# ============================================================
#
# A regime is defined as rare when fewer than 1% of training
# observations belong to that regime.
# ============================================================

regime_df["Regime_Type"] = np.where(
    regime_df[
        "Training_Regime_Frequency"
    ] < 0.01,

    "Rare",

    "Common"
)


# ============================================================
# SUMMARY BY REGIME TYPE
# ============================================================

type_summary = (
    regime_df
    .groupby(
        "Regime_Type"
    )
    .agg(

        Regimes=(
            "Regime",
            "nunique"
        ),

        Test_Observations=(
            "Test_Observations",
            "sum"
        ),

        Mean_MAE=(
            "Mean_Absolute_Error",
            "mean"
        ),

        Mean_RMSE=(
            "RMSE",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# FOCUS ON RARE REGIMES
# ============================================================

rare_regimes = (
    regime_df[
        regime_df[
            "Regime_Type"
        ] == "Rare"
    ]
    .sort_values(
        "Mean_Absolute_Error",
        ascending=False
    )
)


# ============================================================
# PRINT
# ============================================================

print()
print("================================================")
print("FOLD PERFORMANCE")
print("================================================")

print(
    fold_df.to_string(
        index=False
    )
)


print()
print("================================================")
print("REGIME TYPE SUMMARY")
print("================================================")

print(
    type_summary.to_string(
        index=False
    )
)


print()
print("================================================")
print("RARE REGIMES")
print("================================================")

if len(rare_regimes) == 0:

    print(
        "No regimes occurred below the 1% "
        "training-frequency threshold."
    )

else:

    print(
        rare_regimes[
            [
                "Fold",
                "Regime",
                "Test_Observations",
                "Training_Regime_Frequency",
                "Mean_Absolute_Error"
            ]
        ]
        .head(25)
        .to_string(
            index=False
        )
    )


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_ClimateRegime_FoldResults.csv",
    index=False
)

regime_df.to_csv(
    "YCEO_ClimateRegime_AllResults.csv",
    index=False
)

type_summary.to_csv(
    "YCEO_ClimateRegime_TypeSummary.csv",
    index=False
)

rare_regimes.to_csv(
    "YCEO_ClimateRegime_RareRegimes.csv",
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
    "YCEO_ClimateRegime_FoldResults.csv"
)

print(
    "YCEO_ClimateRegime_AllResults.csv"
)

print(
    "YCEO_ClimateRegime_TypeSummary.csv"
)

print(
    "YCEO_ClimateRegime_RareRegimes.csv"
)

print()
print(
    "Climate-regime benchmark complete."
)