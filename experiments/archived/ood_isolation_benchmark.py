import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CODE_DIR = PROJECT_ROOT / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils.paths import DATA_DIR, OOD_RESULTS_DIR
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = DATA_DIR / "YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

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
    # XGBOOST
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


    # ========================================================
    # STANDARDIZE USING TRAINING DATA ONLY
    # ========================================================

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    test_scaled = scaler.transform(
        X_test
    )


    # ========================================================
    # ISOLATION FOREST
    # ========================================================
    #
    # Fit ONLY on training observations.
    #
    # Higher anomaly score = more unusual relative to
    # the training predictor distribution.
    # ========================================================

    detector = IsolationForest(

        n_estimators=300,

        contamination="auto",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    detector.fit(
        train_scaled
    )


    # sklearn's decision_function:
    # higher = more normal.
    #
    # Therefore negate it so that:
    # higher OOD score = more unusual.
    #

    test_ood_score = -detector.decision_function(
        test_scaled
    )


    # ========================================================
    # ABSOLUTE ERROR
    # ========================================================

    absolute_error = np.abs(
        y_test.to_numpy()
        - prediction
    )


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

    meta["Predicted"] = (
        prediction
    )

    meta["Absolute_Error"] = (
        absolute_error
    )

    meta["OOD_Score"] = (
        test_ood_score
    )

    observation_results.append(
        meta
    )


    fold_results.append({

        "Fold": fold,

        "R2": r2,

        "RMSE": rmse,

        "MAE": mae
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

pearson_r, pearson_p = pearsonr(
    obs_df["OOD_Score"],
    obs_df["Absolute_Error"]
)

spearman_r, spearman_p = spearmanr(
    obs_df["OOD_Score"],
    obs_df["Absolute_Error"]
)


# ============================================================
# GLOBAL OOD QUARTILES
# ============================================================

obs_df["OOD_Quartile"] = pd.qcut(
    obs_df["OOD_Score"],
    q=4,
    labels=[
        "Q1 - most typical",
        "Q2",
        "Q3",
        "Q4 - most unusual"
    ],
    duplicates="drop"
)


ood_summary = (
    obs_df
    .groupby(
        "OOD_Quartile",
        observed=True
    )
    .agg(

        Observations=(
            "Code",
            "count"
        ),

        Mean_OOD_Score=(
            "OOD_Score",
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
# WITHIN-FOLD OOD CORRELATIONS
# ============================================================

within_fold = []


for fold in sorted(
    obs_df["Fold"].unique()
):

    temp = obs_df[
        obs_df["Fold"] == fold
    ]

    p_r, p_p = pearsonr(
        temp["OOD_Score"],
        temp["Absolute_Error"]
    )

    s_r, s_p = spearmanr(
        temp["OOD_Score"],
        temp["Absolute_Error"]
    )


    within_fold.append({

        "Fold": fold,

        "Observations": len(temp),

        "Pearson_r": p_r,

        "Pearson_p": p_p,

        "Spearman_r": s_r,

        "Spearman_p": s_p
    })


within_fold_df = pd.DataFrame(
    within_fold
)


# ============================================================
# Q4-Q1 DEGRADATION
# ============================================================

q1 = ood_summary.iloc[0]
q4 = ood_summary.iloc[-1]


degradation = (
    (
        q4["Mean_Absolute_Error"]
        -
        q1["Mean_Absolute_Error"]
    )
    /
    q1["Mean_Absolute_Error"]
    * 100
)


# ============================================================
# PRINT
# ============================================================

print()
print("================================================")
print("OOD QUARTILE PERFORMANCE")
print("================================================")

print(
    ood_summary.to_string(
        index=False
    )
)


print()
print("================================================")
print("OVERALL OOD / ERROR ASSOCIATION")
print("================================================")

print(
    "Pearson r :",
    round(
        pearson_r,
        6
    )
)

print(
    "Pearson p :",
    pearson_p
)

print(
    "Spearman r:",
    round(
        spearman_r,
        6
    )
)

print(
    "Spearman p:",
    spearman_p
)


print()
print("================================================")
print("WITHIN-FOLD OOD CORRELATIONS")
print("================================================")

print(
    within_fold_df.to_string(
        index=False
    )
)


print()
print("================================================")
print("Q4 VS Q1 OOD DEGRADATION")
print("================================================")

print(
    "MAE degradation:",
    round(
        degradation,
        4
    ),
    "%"
)


print()
print("================================================")
print("FIVE-FOLD PERFORMANCE")
print("================================================")

print(
    "R² mean ± SD:",
    round(
        fold_df["R2"].mean(),
        6
    ),
    "±",
    round(
        fold_df["R2"].std(
            ddof=1
        ),
        6
    )
)

print(
    "RMSE mean ± SD:",
    round(
        fold_df["RMSE"].mean(),
        6
    ),
    "±",
    round(
        fold_df["RMSE"].std(
            ddof=1
        ),
        6
    )
)

print(
    "MAE mean ± SD:",
    round(
        fold_df["MAE"].mean(),
        6
    ),
    "±",
    round(
        fold_df["MAE"].std(
            ddof=1
        ),
        6
    )
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(OOD_RESULTS_DIR / "YCEO_OOD_FoldResults.csv",
    index=False
)

obs_df.to_csv(OOD_RESULTS_DIR / "YCEO_OOD_ObservationResults.csv",
    index=False
)

ood_summary.to_csv(OOD_RESULTS_DIR / "YCEO_OOD_ErrorByQuartile.csv",
    index=False
)

within_fold_df.to_csv(OOD_RESULTS_DIR / "YCEO_OOD_WithinFold_Correlations.csv",
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
    "YCEO_OOD_FoldResults.csv"
)

print(
    "YCEO_OOD_ObservationResults.csv"
)

print(
    "YCEO_OOD_ErrorByQuartile.csv"
)

print(
    "YCEO_OOD_WithinFold_Correlations.csv"
)

print()
print(
    "OOD isolation benchmark complete."
)
