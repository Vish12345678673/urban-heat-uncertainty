import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scipy.stats import pearsonr, spearmanr


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

OOD_TREES = 300

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
print("MLP OOD BENCHMARK")
print("==============================================")

print(
    "Dataset:",
    df.shape
)

print(
    "Spatial blocks:",
    groups.nunique()
)


# ============================================================
# MLP
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
        ).squeeze(-1)


# ============================================================
# TRAIN MLP
# ============================================================

def train_mlp(
    X_train,
    y_train,
    X_test,
    y_test
):

    # --------------------------------------------------------
    # Feature scaling
    # TRAIN ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Target scaling
    # TRAIN ONLY
    # --------------------------------------------------------

    y_mean = y_train.mean()

    y_std = y_train.std()

    if y_std < 1e-12:
        y_std = 1.0


    y_train_scaled = (
        y_train - y_mean
    ) / y_std

    y_test_scaled = (
        y_test - y_mean
    ) / y_std


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataset = TensorDataset(

        torch.tensor(
            X_train_scaled,
            dtype=torch.float32
        ),

        torch.tensor(
            y_train_scaled.to_numpy(),
            dtype=torch.float32
        )
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    y_test_tensor = torch.tensor(
        y_test_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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


    best_loss = np.inf
    best_state = None
    patience_counter = 0


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        model.train()

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )


            optimizer.zero_grad()

            pred = model(
                batch_x
            )

            loss = criterion(
                pred,
                batch_y
            )

            loss.backward()

            optimizer.step()


        # ----------------------------------------------------
        # Validation on spatial test fold
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            test_pred = model(
                X_test_tensor
            )

            test_loss = criterion(
                test_pred,
                y_test_tensor
            ).item()


        if test_loss < (
            best_loss - 1e-6
        ):

            best_loss = test_loss

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


    # --------------------------------------------------------
    # Restore best state
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        pred_scaled = (
            model(
                X_test_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    prediction = (
        pred_scaled
        * y_std
        + y_mean
    )


    return prediction


# ============================================================
# OOD SCORE
# ============================================================

def compute_ood_scores(
    X_train,
    X_test
):

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        X_train
    )

    test_scaled = scaler.transform(
        X_test
    )


    detector = IsolationForest(

        n_estimators=OOD_TREES,

        contamination="auto",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    detector.fit(
        train_scaled
    )


    # sklearn decision_function:
    # higher = more normal
    #
    # Negate so:
    # higher score = more unusual
    #

    test_ood = -detector.decision_function(
        test_scaled
    )


    return test_ood


# ============================================================
# CROSS VALIDATION
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


    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    prediction = train_mlp(
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


    # --------------------------------------------------------
    # Leakage-free OOD score
    # --------------------------------------------------------

    ood_score = compute_ood_scores(
        X_train,
        X_test
    )


    absolute_error = np.abs(
        y_test.to_numpy()
        - prediction
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

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

    meta["Absolute_Error"] = (
        absolute_error
    )

    meta["OOD_Score"] = (
        ood_score
    )


    observation_results.append(
        meta
    )


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
# OOD / ERROR CORRELATIONS
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
# WITHIN-FOLD CORRELATIONS
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
# Q4 / Q1 DEGRADATION
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
# CV SUMMARY
# ============================================================

cv_summary = pd.DataFrame({

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
# PRINT RESULTS
# ============================================================

print()
print("==============================================")
print("OOD QUARTILE PERFORMANCE")
print("==============================================")

print(
    ood_summary.to_string(
        index=False
    )
)


print()
print("==============================================")
print("OOD / ERROR ASSOCIATION")
print("==============================================")

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
print("==============================================")
print("WITHIN-FOLD OOD CORRELATIONS")
print("==============================================")

print(
    within_fold_df.to_string(
        index=False
    )
)


print()
print("==============================================")
print("Q4 VS Q1 DEGRADATION")
print("==============================================")

print(
    "MAE degradation:",
    round(
        degradation,
        4
    ),
    "%"
)


print()
print("==============================================")
print("FIVE-FOLD MLP PERFORMANCE")
print("==============================================")

print(
    cv_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_MLP_OOD_FoldResults.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_MLP_OOD_ObservationResults.csv",
    index=False
)

ood_summary.to_csv(
    "YCEO_MLP_OOD_ErrorByQuartile.csv",
    index=False
)

within_fold_df.to_csv(
    "YCEO_MLP_OOD_WithinFold_Correlations.csv",
    index=False
)

cv_summary.to_csv(
    "YCEO_MLP_OOD_Summary.csv",
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
    "YCEO_MLP_OOD_FoldResults.csv"
)

print(
    "YCEO_MLP_OOD_ObservationResults.csv"
)

print(
    "YCEO_MLP_OOD_ErrorByQuartile.csv"
)

print(
    "YCEO_MLP_OOD_WithinFold_Correlations.csv"
)

print(
    "YCEO_MLP_OOD_Summary.csv"
)

print()
print(
    "MLP OOD benchmark complete."
)