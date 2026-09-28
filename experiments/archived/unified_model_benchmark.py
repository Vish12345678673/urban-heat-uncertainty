import os
import random
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor

import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42

N_OUTER_FOLDS = 5

INTERNAL_VAL_SIZE = 0.15

EPOCHS = 200

BATCH_SIZE = 128

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

HIDDEN_1 = 128

HIDDEN_2 = 64

DROPOUT = 0.10

PATIENCE = 20

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
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD AND CLEAN
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
print("UNIFIED LEAKAGE-FREE MODEL BENCHMARK")
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
    "Spatial blocks:",
    groups.nunique()
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# XGBOOST
# ============================================================

def make_xgb():

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
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                HIDDEN_1
            ),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_1,
                HIDDEN_2
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                HIDDEN_2
            ),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_2,
                1
            )
        )


    def forward(self, x):

        return self.net(
            x
        ).squeeze(-1)


# ============================================================
# TRAIN MLP WITH INTERNAL SPATIAL VALIDATION
# ============================================================

def train_mlp(
    X_outer_train,
    y_outer_train
):

    # --------------------------------------------------------
    # Internal spatial split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,

        test_size=INTERNAL_VAL_SIZE,

        random_state=RANDOM_STATE
    )


    inner_lat = np.floor(
        X_outer_train["_Lat_"] / 10
    ).astype(int)

    inner_lon = np.floor(
        X_outer_train["_Lon_"] / 10
    ).astype(int)


    inner_groups = (
        inner_lat.astype(str)
        + "_"
        + inner_lon.astype(str)
    )


    inner_train_idx, inner_val_idx = next(

        splitter.split(
            X_outer_train,

            y_outer_train,

            groups=inner_groups
        )
    )


    X_train = X_outer_train.iloc[
        inner_train_idx
    ][FEATURES]

    X_val = X_outer_train.iloc[
        inner_val_idx
    ][FEATURES]

    y_train = y_outer_train.iloc[
        inner_train_idx
    ]

    y_val = y_outer_train.iloc[
        inner_val_idx
    ]


    # --------------------------------------------------------
    # Scale using internal training only
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )


    # --------------------------------------------------------
    # Target scaling
    # --------------------------------------------------------

    y_mean = y_train.mean()

    y_std = y_train.std()

    if y_std < 1e-12:

        y_std = 1.0


    y_train_scaled = (
        y_train
        - y_mean
    ) / y_std


    y_val_scaled = (
        y_val
        - y_mean
    ) / y_std


    # --------------------------------------------------------
    # Tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(

        X_train_scaled,

        dtype=torch.float32,

        device=DEVICE
    )

    y_train_tensor = torch.tensor(

        y_train_scaled.to_numpy(),

        dtype=torch.float32,

        device=DEVICE
    )

    X_val_tensor = torch.tensor(

        X_val_scaled,

        dtype=torch.float32,

        device=DEVICE
    )

    y_val_tensor = torch.tensor(

        y_val_scaled.to_numpy(),

        dtype=torch.float32,

        device=DEVICE
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MLP(
        len(FEATURES)
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )


    criterion = nn.MSELoss()


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_loss = np.inf

    best_state = None

    patience_counter = 0


    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        prediction = model(
            X_train_tensor
        )

        loss = criterion(
            prediction,
            y_train_tensor
        )

        loss.backward()

        optimizer.step()


        model.eval()

        with torch.no_grad():

            val_prediction = model(
                X_val_tensor
            )

            val_loss = criterion(
                val_prediction,
                y_val_tensor
            ).item()


        if val_loss < (
            best_loss
            - 1e-6
        ):

            best_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()

                for k, v
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:

            break


    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    return (
        model,
        scaler,
        y_mean,
        y_std
    )


# ============================================================
# OUTER CV
# ============================================================

X_with_coords = X.copy()

X_with_coords["_Lat_"] = df["Lat"].values

X_with_coords["_Lon_"] = df["Lon"].values


outer_cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)


xgb_results = []

mlp_results = []

observation_results = []


# ============================================================
# RUN SAME OUTER FOLDS FOR BOTH MODELS
# ============================================================

for fold, (
    train_idx,
    test_idx
) in enumerate(

    outer_cv.split(
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
    # XGBOOST
    # ========================================================

    start_time = time.time()

    xgb = make_xgb()

    xgb.fit(
        X_train,
        y_train
    )

    xgb_pred = xgb.predict(
        X_test
    )

    xgb_time = (
        time.time()
        - start_time
    )


    # ========================================================
    # MLP
    # ========================================================

    start_time = time.time()


    X_outer_train = X_with_coords.iloc[
        train_idx
    ].copy()


    (
        mlp,
        mlp_scaler,
        mlp_y_mean,
        mlp_y_std
    ) = train_mlp(

        X_outer_train,

        y_train
    )


    X_test_scaled = mlp_scaler.transform(
        X_test
    )


    X_test_tensor = torch.tensor(

        X_test_scaled,

        dtype=torch.float32,

        device=DEVICE
    )


    mlp.eval()

    with torch.no_grad():

        mlp_pred_scaled = (
            mlp(
                X_test_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    mlp_pred = (
        mlp_pred_scaled
        * mlp_y_std
        + mlp_y_mean
    )


    mlp_time = (
        time.time()
        - start_time
    )


    # ========================================================
    # METRICS
    # ========================================================

    xgb_r2 = r2_score(
        y_test,
        xgb_pred
    )

    xgb_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            xgb_pred
        )
    )

    xgb_mae = mean_absolute_error(
        y_test,
        xgb_pred
    )


    mlp_r2 = r2_score(
        y_test,
        mlp_pred
    )

    mlp_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            mlp_pred
        )
    )

    mlp_mae = mean_absolute_error(
        y_test,
        mlp_pred
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("XGBOOST")

    print(
        "R²   =",
        round(xgb_r2, 6)
    )

    print(
        "RMSE =",
        round(xgb_rmse, 6)
    )

    print(
        "MAE  =",
        round(xgb_mae, 6)
    )


    print()
    print("LEAKAGE-FREE MLP")

    print(
        "R²   =",
        round(mlp_r2, 6)
    )

    print(
        "RMSE =",
        round(mlp_rmse, 6)
    )

    print(
        "MAE  =",
        round(mlp_mae, 6)
    )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    xgb_results.append({

        "Fold": fold,

        "Model": "XGBoost",

        "R2": xgb_r2,

        "RMSE": xgb_rmse,

        "MAE": xgb_mae,

        "Training_Time_Seconds":
            xgb_time

    })


    mlp_results.append({

        "Fold": fold,

        "Model": "Leakage_Free_MLP",

        "R2": mlp_r2,

        "RMSE": mlp_rmse,

        "MAE": mlp_mae,

        "Training_Time_Seconds":
            mlp_time

    })


    # ========================================================
    # OBSERVATIONS
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

    meta["XGBoost_Predicted"] = (
        xgb_pred
    )

    meta["XGBoost_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - xgb_pred
        )
    )

    meta["MLP_Predicted"] = (
        mlp_pred
    )

    meta["MLP_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - mlp_pred
        )
    )


    observation_results.append(
        meta
    )


# ============================================================
# COMBINE
# ============================================================

xgb_df = pd.DataFrame(
    xgb_results
)

mlp_df = pd.DataFrame(
    mlp_results
)

comparison_df = pd.concat(
    [
        xgb_df,
        mlp_df
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
        ),

        Mean_Training_Time=(
            "Training_Time_Seconds",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# PAIRED MODEL DIFFERENCES
# ============================================================

paired = (
    comparison_df
    .pivot(
        index="Fold",
        columns="Model",
        values=[
            "R2",
            "RMSE",
            "MAE"
        ]
    )
)


paired_rows = []


for fold in sorted(
    comparison_df["Fold"].unique()
):

    xgb_row = xgb_df[
        xgb_df["Fold"] == fold
    ].iloc[0]

    mlp_row = mlp_df[
        mlp_df["Fold"] == fold
    ].iloc[0]


    paired_rows.append({

        "Fold": fold,

        "Delta_R2_MLP_minus_XGB":
            mlp_row["R2"]
            - xgb_row["R2"],

        "Delta_RMSE_MLP_minus_XGB":
            mlp_row["RMSE"]
            - xgb_row["RMSE"],

        "Delta_MAE_MLP_minus_XGB":
            mlp_row["MAE"]
            - xgb_row["MAE"]
    })


paired_df = pd.DataFrame(
    paired_rows
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("============================================================")
print("UNIFIED FINAL SUMMARY")
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
    "YCEO_Unified_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_Unified_Model_Summary.csv",
    index=False
)

paired_df.to_csv(
    "YCEO_Unified_Paired_Differences.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_Unified_Observation_Results.csv",
    index=False
)


print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_Unified_Model_Comparison.csv"
)

print(
    "YCEO_Unified_Model_Summary.csv"
)

print(
    "YCEO_Unified_Paired_Differences.csv"
)

print(
    "YCEO_Unified_Observation_Results.csv"
)

print()
print(
    "Unified model benchmark complete."
)