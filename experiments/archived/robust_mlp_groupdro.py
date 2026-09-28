import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

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
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_1 = 128
HIDDEN_2 = 64
DROPOUT = 0.10

PATIENCE = 20

OOD_TREES = 300

# Robustness strength.
# 0.0 = ordinary mean loss
# 1.0 = strongest worst-group emphasis
ROBUST_ALPHA = 0.50

# Temperature for smooth maximum.
ROBUST_TAU = 0.10

N_OOD_GROUPS = 4


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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


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
print("==============================================")
print("OOD-ROBUST MLP / GROUP-DRO")
print("==============================================")

print("Dataset:", df.shape)
print("Spatial blocks:", groups.nunique())
print("Features:", len(FEATURES))


# ============================================================
# MODEL
# ============================================================

class RobustMLP(nn.Module):

    def __init__(self, input_dim):

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

    def forward(self, x):

        return self.network(
            x
        ).squeeze(-1)


# ============================================================
# OOD SCORE
# ============================================================

def compute_ood_scores(
    X_train
):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    detector = IsolationForest(

        n_estimators=OOD_TREES,

        contamination="auto",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    detector.fit(
        X_scaled
    )

    # Higher = more unusual.
    score = -detector.decision_function(
        X_scaled
    )

    return score


# ============================================================
# BUILD GROUPS FROM TRAINING DATA ONLY
# ============================================================

def build_ood_groups(
    X_train
):

    score = compute_ood_scores(
        X_train
    )

    # Quantile groups.
    group = pd.qcut(
        score,
        q=N_OOD_GROUPS,
        labels=False,
        duplicates="drop"
    )

    group = np.asarray(
        group,
        dtype=int
    )

    return group, score


# ============================================================
# TRAIN ROBUST MLP
# ============================================================

def train_robust_mlp(
    X_outer_train,
    y_outer_train
):

    # --------------------------------------------------------
    # Internal spatial validation
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
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )


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


    # --------------------------------------------------------
    # TRAINING OOD GROUPS
    # --------------------------------------------------------

    ood_groups, ood_scores = build_ood_groups(
        X_train
    )


    n_groups = int(
        np.max(ood_groups)
    ) + 1


    # --------------------------------------------------------
    # tensors
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

    group_tensor = torch.tensor(
        ood_groups,
        dtype=torch.long,
        device=DEVICE
    )


    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = RobustMLP(
        input_dim=len(FEATURES)
    ).to(DEVICE)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    mse = nn.MSELoss()


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

        optimizer.zero_grad()

        prediction = model(
            X_train_tensor
        )


        # ----------------------------------------------------
        # Per-group training losses
        # ----------------------------------------------------

        group_losses = []


        for group_id in range(
            n_groups
        ):

            mask = (
                group_tensor
                == group_id
            )

            if mask.sum() == 0:
                continue


            group_loss = torch.mean(
                (
                    prediction[mask]
                    -
                    y_train_tensor[mask]
                ) ** 2
            )

            group_losses.append(
                group_loss
            )


        group_losses_tensor = torch.stack(
            group_losses
        )


        # ----------------------------------------------------
        # Mean risk
        # ----------------------------------------------------

        mean_loss = torch.mean(
            group_losses_tensor
        )


        # ----------------------------------------------------
        # Smooth worst-group risk
        # ----------------------------------------------------

        worst_group_loss = (
            ROBUST_TAU
            *
            torch.logsumexp(
                group_losses_tensor
                / ROBUST_TAU,
                dim=0
            )
        )


        # ----------------------------------------------------
        # Combined objective
        # ----------------------------------------------------

        loss = (
            (1.0 - ROBUST_ALPHA)
            * mean_loss
            +
            ROBUST_ALPHA
            * worst_group_loss
        )


        loss.backward()

        optimizer.step()


        # ====================================================
        # INTERNAL VALIDATION
        # ====================================================

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                X_val_tensor
            )

            val_loss = mse(
                val_prediction,
                y_val_tensor
            ).item()


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < (
            best_val_loss - 1e-6
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
    # RESTORE BEST MODEL
    # ========================================================

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
# STANDARD MLP BASELINE
# ============================================================

def train_standard_mlp(
    X_outer_train,
    y_outer_train
):

    # --------------------------------------------------------
    # Internal spatial validation
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


    train_idx, val_idx = next(
        splitter.split(
            X_outer_train,
            y_outer_train,
            groups=inner_groups
        )
    )


    X_train = X_outer_train.iloc[
        train_idx
    ][FEATURES]

    X_val = X_outer_train.iloc[
        val_idx
    ][FEATURES]

    y_train = y_outer_train.iloc[
        train_idx
    ]

    y_val = y_outer_train.iloc[
        val_idx
    ]


    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )


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


    train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    target_tensor = torch.tensor(
        y_train_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )

    val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    val_target = torch.tensor(
        y_val_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )


    model = RobustMLP(
        input_dim=len(FEATURES)
    ).to(DEVICE)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()


    best_loss = np.inf
    best_state = None
    patience_counter = 0


    for epoch in range(
        EPOCHS
    ):

        model.train()

        optimizer.zero_grad()

        prediction = model(
            train_tensor
        )

        loss = criterion(
            prediction,
            target_tensor
        )

        loss.backward()

        optimizer.step()


        model.eval()

        with torch.no_grad():

            val_prediction = model(
                val_tensor
            )

            val_loss = criterion(
                val_prediction,
                val_target
            ).item()


        if val_loss < (
            best_loss - 1e-6
        ):

            best_loss = val_loss

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
# OUTER SPATIAL CV
# ============================================================

outer_cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)

baseline_results = []
robust_results = []

observation_results = []


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
    print("================================================")
    print("OUTER SPATIAL FOLD", fold)
    print("================================================")


    X_outer_train = X_with_coords = X.copy()

    X_outer_train = X.copy()

    X_train_coords = X_outer_train.copy()

    X_train_coords["_Lat_"] = df["Lat"].values
    X_train_coords["_Lon_"] = df["Lon"].values


    X_train_outer = X_train_coords.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_train_outer = y.iloc[
        train_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    # ========================================================
    # STANDARD MLP
    # ========================================================

    (
        baseline_model,
        baseline_scaler,
        baseline_y_mean,
        baseline_y_std
    ) = train_standard_mlp(
        X_train_outer,
        y_train_outer
    )


    X_test_scaled = baseline_scaler.transform(
        X_test
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    baseline_model.eval()

    with torch.no_grad():

        baseline_scaled = (
            baseline_model(
                X_test_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    baseline_pred = (
        baseline_scaled
        * baseline_y_std
        + baseline_y_mean
    )


    # ========================================================
    # ROBUST MLP
    # ========================================================

    (
        robust_model,
        robust_scaler,
        robust_y_mean,
        robust_y_std
    ) = train_robust_mlp(
        X_train_outer,
        y_train_outer
    )


    X_test_scaled_robust = robust_scaler.transform(
        X_test
    )

    X_test_tensor_robust = torch.tensor(
        X_test_scaled_robust,
        dtype=torch.float32,
        device=DEVICE
    )


    robust_model.eval()

    with torch.no_grad():

        robust_scaled = (
            robust_model(
                X_test_tensor_robust
            )
            .detach()
            .cpu()
            .numpy()
        )


    robust_pred = (
        robust_scaled
        * robust_y_std
        + robust_y_mean
    )


    # ========================================================
    # METRICS
    # ========================================================

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


    print()
    print("STANDARD MLP")

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
    print("GROUP-DRO MLP")

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
    # STORE FOLD RESULTS
    # ========================================================

    baseline_results.append({

        "Fold": fold,

        "Model": "Standard_MLP",

        "R2": baseline_r2,

        "RMSE": baseline_rmse,

        "MAE": baseline_mae
    })


    robust_results.append({

        "Fold": fold,

        "Model": "Group_DRO_MLP",

        "R2": robust_r2,

        "RMSE": robust_rmse,

        "MAE": robust_mae
    })


    # ========================================================
    # OUTER TEST OOD DETECTOR
    # ========================================================

    test_scaler_ood = StandardScaler()

    outer_train_scaled = (
        test_scaler_ood.fit_transform(
            X_train_outer[FEATURES]
        )
    )

    outer_test_scaled = (
        test_scaler_ood.transform(
            X_test[FEATURES]
        )
    )


    detector = IsolationForest(

        n_estimators=OOD_TREES,

        contamination="auto",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    detector.fit(
        outer_train_scaled
    )


    test_ood = -detector.decision_function(
        outer_test_scaled
    )


    # --------------------------------------------------------
    # Observation records
    # --------------------------------------------------------

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

    meta["OOD_Score"] = (
        test_ood
    )

    meta["Standard_MLP_Predicted"] = (
        baseline_pred
    )

    meta["Standard_MLP_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - baseline_pred
        )
    )

    meta["GroupDRO_MLP_Predicted"] = (
        robust_pred
    )

    meta["GroupDRO_MLP_Absolute_Error"] = (
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
# OOD QUARTILES
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

        Standard_MLP_MAE=(
            "Standard_MLP_Absolute_Error",
            "mean"
        ),

        GroupDRO_MLP_MAE=(
            "GroupDRO_MLP_Absolute_Error",
            "mean"
        )
    )
    .reset_index()
)


# ============================================================
# Q1/Q4 DEGRADATION
# ============================================================

q1 = ood_summary.iloc[0]
q4 = ood_summary.iloc[-1]


standard_degradation = (

    (
        q4["Standard_MLP_MAE"]
        -
        q1["Standard_MLP_MAE"]
    )

    /
    q1["Standard_MLP_MAE"]

    * 100
)


robust_degradation = (

    (
        q4["GroupDRO_MLP_MAE"]
        -
        q1["GroupDRO_MLP_MAE"]
    )

    /
    q1["GroupDRO_MLP_MAE"]

    * 100
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


    standard_corr = np.corrcoef(
        temp["OOD_Score"],
        temp["Standard_MLP_Absolute_Error"]
    )[0, 1]


    robust_corr = np.corrcoef(
        temp["OOD_Score"],
        temp["GroupDRO_MLP_Absolute_Error"]
    )[0, 1]


    within_fold.append({

        "Fold": fold,

        "Standard_MLP_OOD_Error_r":
            standard_corr,

        "GroupDRO_MLP_OOD_Error_r":
            robust_corr
    })


within_fold_df = pd.DataFrame(
    within_fold
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
# PRINT
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
print("OOD QUARTILE PERFORMANCE")
print("================================================")

print(
    ood_summary.to_string(
        index=False
    )
)


print()
print("================================================")
print("Q4 VS Q1 OOD DEGRADATION")
print("================================================")

print(
    "Standard MLP:",
    round(
        standard_degradation,
        4
    ),
    "%"
)

print(
    "Group-DRO MLP:",
    round(
        robust_degradation,
        4
    ),
    "%"
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


# ============================================================
# SAVE
# ============================================================

comparison_df.to_csv(
    "YCEO_GroupDRO_MLP_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_GroupDRO_MLP_Summary.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_GroupDRO_MLP_Observation_Results.csv",
    index=False
)

ood_summary.to_csv(
    "YCEO_GroupDRO_MLP_OOD_Quartiles.csv",
    index=False
)

within_fold_df.to_csv(
    "YCEO_GroupDRO_MLP_WithinFold_OOD.csv",
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
    "YCEO_GroupDRO_MLP_Comparison.csv"
)

print(
    "YCEO_GroupDRO_MLP_Summary.csv"
)

print(
    "YCEO_GroupDRO_MLP_Observation_Results.csv"
)

print(
    "YCEO_GroupDRO_MLP_OOD_Quartiles.csv"
)

print(
    "YCEO_GroupDRO_MLP_WithinFold_OOD.csv"
)

print()
print(
    "Group-DRO OOD-robust MLP complete."
)