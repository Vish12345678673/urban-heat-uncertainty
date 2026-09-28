import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
N_FOLDS = 5

EPOCHS = 150
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

random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)

torch.manual_seed(
    RANDOM_STATE
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        RANDOM_STATE
    )

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    DEVICE
)


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


X = df[FEATURES].copy()
y = df[TARGET].copy()
groups = df["spatial_block"].copy()


print()
print("==============================================")
print("MLP SPATIAL BASELINE")
print("==============================================")

print(
    "Dataset:",
    df.shape
)

print(
    "Spatial blocks:",
    groups.nunique()
)

print(
    "Features:",
    len(FEATURES)
)


# ============================================================
# MODEL
# ============================================================

class MLPRegressor(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

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


    def forward(
        self,
        x
    ):

        return self.network(
            x
        ).squeeze(
            -1
        )


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val
):

    scaler_x = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(
        X_train
    )

    X_val_scaled = scaler_x.transform(
        X_val
    )


    # Scale target using TRAINING data only.
    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-12:
        y_std = 1.0

    y_train_scaled = (
        y_train - y_mean
    ) / y_std

    y_val_scaled = (
        y_val - y_mean
    ) / y_std


    train_dataset = TensorDataset(

        torch.tensor(
            X_train_scaled,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train_scaled.to_numpy(),
            dtype=torch.float32
        )
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
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


    model = MLPRegressor(
        input_dim=len(FEATURES)
    ).to(
        DEVICE
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()


    best_val_loss = np.inf
    best_state = None
    patience_counter = 0


    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0
        total_count = 0


        for batch_x, batch_y in train_loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )


            optimizer.zero_grad()

            prediction = model(
                batch_x
            )

            loss = criterion(
                prediction,
                batch_y
            )

            loss.backward()

            optimizer.step()


            n = len(
                batch_x
            )

            total_loss += (
                loss.item()
                * n
            )

            total_count += n


        train_loss = (
            total_loss
            / max(
                total_count,
                1
            )
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                X_val_tensor
            )

            val_loss = criterion(
                val_prediction,
                y_val_tensor
            ).item()


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < (
            best_val_loss
            - 1e-6
        ):

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:

            break


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    model.eval()

    with torch.no_grad():

        prediction_scaled = (
            model(
                X_val_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    # Convert back to original target units.
    prediction = (
        prediction_scaled
        * y_std
        + y_mean
    )


    return (
        prediction,
        model,
        scaler_x
    )


# ============================================================
# FIVE-FOLD SPATIAL CV
# ============================================================

cv = GroupKFold(
    n_splits=N_FOLDS
)

fold_results = []
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
    print("----------------------------------------------")
    print(
        "SPATIAL FOLD",
        fold
    )
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


    prediction, model, scaler = train_model(
        X_train,
        y_train,
        X_test,
        y_test
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

    meta["Absolute_Error"] = np.abs(
        y_test.to_numpy()
        - prediction
    )

    observation_results.append(
        meta
    )


# ============================================================
# RESULTS
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


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
# PRINT
# ============================================================

print()
print("==============================================")
print("FIVE-FOLD SPATIAL CV")
print("==============================================")

print(
    fold_df.to_string(
        index=False
    )
)


print()
print("==============================================")
print("MEAN ± SD")
print("==============================================")

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_MLP_Spatial_CV.csv",
    index=False
)

summary.to_csv(
    "YCEO_MLP_Spatial_CV_Summary.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_MLP_Spatial_Observation_Results.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==============================================")
print("FILES CREATED")
print("==============================================")

print(
    "YCEO_MLP_Spatial_CV.csv"
)

print(
    "YCEO_MLP_Spatial_CV_Summary.csv"
)

print(
    "YCEO_MLP_Spatial_Observation_Results.csv"
)

print()
print(
    "MLP spatial baseline complete."
)