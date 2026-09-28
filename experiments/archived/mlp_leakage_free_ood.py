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

from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
print("LEAKAGE-FREE MLP OOD BENCHMARK")
print("==============================================")

print("Dataset:", df.shape)
print("Spatial blocks:", groups.nunique())
print("Features:", len(FEATURES))


# ============================================================
# MLP
# ============================================================

class MLPRegressor(nn.Module):

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
# TRAIN MLP USING INTERNAL VALIDATION ONLY
# ============================================================

def train_mlp(
    X_outer_train,
    y_outer_train
):

    # --------------------------------------------------------
    # Internal spatial validation split
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
    # Feature scaler: TRAIN ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )


    # --------------------------------------------------------
    # Target scaling: TRAIN ONLY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = MLPRegressor(
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


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

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


        # ----------------------------------------------------
        # INTERNAL VALIDATION ONLY
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


    # --------------------------------------------------------
    # RESTORE BEST INTERNAL MODEL
    # --------------------------------------------------------

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
# ADD COORDINATES FOR INTERNAL GROUPING
# ============================================================

X_with_coords = X.copy()

X_with_coords["_Lat_"] = df["Lat"].values
X_with_coords["_Lon_"] = df["Lon"].values


# ============================================================
# OUTER SPATIAL CV
# ============================================================

outer_cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)

fold_results = []
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
    print("----------------------------------------------")
    print(
        "OUTER SPATIAL FOLD",
        fold
    )
    print("----------------------------------------------")


    X_outer_train = X_with_coords.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_outer_train = y.iloc[
        train_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    # ========================================================
    # TRAIN ONLY ON OUTER TRAINING DATA
    # ========================================================

    (
        model,
        scaler,
        y_mean,
        y_std
    ) = train_mlp(
        X_outer_train,
        y_outer_train
    )


    # ========================================================
    # TEST PREDICTION
    # ========================================================

    X_test_scaled = scaler.transform(
        X_test
    )

    X_test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    model.eval()

    with torch.no_grad():

        prediction_scaled = (
            model(
                X_test_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    prediction = (
        prediction_scaled
        * y_std
        + y_mean
    )


    # ========================================================
    # PREDICTION METRICS
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
    # LEAKAGE-FREE OOD DETECTOR
    # ========================================================

    train_scaler_ood = StandardScaler()

    train_scaled_ood = (
        train_scaler_ood.fit_transform(
            X_outer_train[FEATURES]
        )
    )

    test_scaled_ood = (
        train_scaler_ood.transform(
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
        train_scaled_ood
    )


    # Higher = more unusual.
    test_ood_score = -detector.decision_function(
        test_scaled_ood
    )


    # ========================================================
    # ERROR
    # ========================================================

    absolute_error = np.abs(
        y_test.to_numpy()
        - prediction
    )


    # ========================================================
    # STORE
    # ========================================================

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
        test_ood_score
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
# OVERALL ASSOCIATIONS
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
# WITHIN-FOLD ASSOCIATIONS
# ============================================================

within_fold = []


for fold in sorted(
    obs_df["Fold"].unique()
):

    temp = obs_df[
        obs_df["Fold"] == fold
    ]


    pr, pp = pearsonr(
        temp["OOD_Score"],
        temp["Absolute_Error"]
    )

    sr, sp = spearmanr(
        temp["OOD_Score"],
        temp["Absolute_Error"]
    )


    within_fold.append({

        "Fold": fold,

        "Observations": len(temp),

        "Pearson_r": pr,

        "Pearson_p": pp,

        "Spearman_r": sr,

        "Spearman_p": sp
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
        - q1["Mean_Absolute_Error"]
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
# PRINT
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
print("OVERALL OOD / ERROR ASSOCIATION")
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
print("LEAKAGE-FREE MLP PERFORMANCE")
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
    "YCEO_MLP_LeakageFree_OOD_FoldResults.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_MLP_LeakageFree_OOD_ObservationResults.csv",
    index=False
)

ood_summary.to_csv(
    "YCEO_MLP_LeakageFree_OOD_ErrorByQuartile.csv",
    index=False
)

within_fold_df.to_csv(
    "YCEO_MLP_LeakageFree_OOD_WithinFold_Correlations.csv",
    index=False
)

cv_summary.to_csv(
    "YCEO_MLP_LeakageFree_OOD_Summary.csv",
    index=False
)


print()
print("==============================================")
print("FILES CREATED")
print("==============================================")

print(
    "YCEO_MLP_LeakageFree_OOD_FoldResults.csv"
)

print(
    "YCEO_MLP_LeakageFree_OOD_ObservationResults.csv"
)

print(
    "YCEO_MLP_LeakageFree_OOD_ErrorByQuartile.csv"
)

print(
    "YCEO_MLP_LeakageFree_OOD_WithinFold_Correlations.csv"
)

print(
    "YCEO_MLP_LeakageFree_OOD_Summary.csv"
)

print()
print(
    "Leakage-free MLP OOD benchmark complete."
)