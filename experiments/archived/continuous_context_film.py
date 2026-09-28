import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

N_CONTEXT_COMPONENTS = 3

EPOCHS = 200

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 64

CONTEXT_DIM = 3

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


# Coordinates kept only for internal spatial validation.
X_with_coords = X.copy()

X_with_coords["_Lat_"] = df["Lat"].values
X_with_coords["_Lon_"] = df["Lon"].values


print()
print("============================================================")
print("CONTINUOUS-CONTEXT FiLM")
print("============================================================")

print("Dataset:", df.shape)
print("Features:", len(FEATURES))
print("Spatial blocks:", groups.nunique())
print("Context dimensions:", CONTEXT_DIM)
print("Device:", DEVICE)


# ============================================================
# STANDARD MLP
# ============================================================

class StandardMLP(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                HIDDEN_DIM
            ),

            nn.Dropout(
                DROPOUT
            )
        )

        self.head = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                1
            )
        )

    def forward(self, x):

        h = self.encoder(x)

        return self.head(h).squeeze(-1)


# ============================================================
# CONTINUOUS-CONTEXT FiLM MODEL
# ============================================================

class ContinuousContextFiLM(nn.Module):

    def __init__(
        self,
        input_dim,
        context_dim
    ):

        super().__init__()

        # ----------------------------------------------------
        # Shared feature encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                HIDDEN_DIM
            ),

            nn.Dropout(
                DROPOUT
            )
        )


        # ----------------------------------------------------
        # Context encoder
        # ----------------------------------------------------

        self.context_encoder = nn.Sequential(

            nn.Linear(
                context_dim,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU()
        )


        # ----------------------------------------------------
        # FiLM parameters
        # ----------------------------------------------------

        self.gamma = nn.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM
        )

        self.beta = nn.Linear(
            HIDDEN_DIM,
            HIDDEN_DIM
        )


        # ----------------------------------------------------
        # Prediction head
        # ----------------------------------------------------

        self.head = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM,
                1
            )
        )


    def forward(
        self,
        x,
        context
    ):

        h = self.encoder(x)

        z = self.context_encoder(
            context
        )

        gamma = self.gamma(z)

        beta = self.beta(z)

        # ----------------------------------------------------
        # Feature-wise Linear Modulation
        # ----------------------------------------------------

        h = (
            (1.0 + gamma)
            * h
            + beta
        )

        h = torch.relu(h)

        prediction = self.head(
            h
        ).squeeze(-1)

        return prediction


# ============================================================
# BUILD CONTINUOUS CONTEXT
# ============================================================

def fit_context_transform(
    X_train
):

    # --------------------------------------------------------
    # Predictor standardization fitted on TRAIN ONLY.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train[FEATURES]
    )


    # --------------------------------------------------------
    # PCA fitted on TRAIN ONLY.
    # --------------------------------------------------------

    n_components = min(
        N_CONTEXT_COMPONENTS,
        len(FEATURES)
    )

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    context = pca.fit_transform(
        X_scaled
    )


    # --------------------------------------------------------
    # Standardize PCA coordinates using training data only.
    #
    # This makes context dimensions numerically comparable.
    # --------------------------------------------------------

    context_scaler = StandardScaler()

    context_scaled = context_scaler.fit_transform(
        context
    )


    return (
        scaler,
        pca,
        context_scaler,
        context_scaled
    )


def transform_context(
    X_data,
    scaler,
    pca,
    context_scaler
):

    X_scaled = scaler.transform(
        X_data[FEATURES]
    )

    context = pca.transform(
        X_scaled
    )

    context_scaled = context_scaler.transform(
        context
    )

    return context_scaled


# ============================================================
# TRAIN STANDARD MLP
# ============================================================

def train_standard_mlp(
    X_outer_train,
    y_outer_train
):

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


    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train
    )

    X_val_scaled = feature_scaler.transform(
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
        y_train - y_mean
    ) / y_std

    y_val_scaled = (
        y_val - y_mean
    ) / y_std


    train_x = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    train_y = torch.tensor(
        y_train_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )

    val_x = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    val_y = torch.tensor(
        y_val_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )


    model = StandardMLP(
        len(FEATURES)
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


    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        prediction = model(
            train_x
        )

        loss = criterion(
            prediction,
            train_y
        )

        loss.backward()

        optimizer.step()


        model.eval()

        with torch.no_grad():

            val_prediction = model(
                val_x
            )

            val_loss = criterion(
                val_prediction,
                val_y
            ).item()


        if val_loss < (
            best_loss - 1e-6
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
        feature_scaler,
        y_mean,
        y_std
    )


# ============================================================
# TRAIN CONTINUOUS-CONTEXT FiLM
# ============================================================

def train_film(
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


    # --------------------------------------------------------
    # Feature scaling
    # --------------------------------------------------------

    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train
    )

    X_val_scaled = feature_scaler.transform(
        X_val
    )


    # --------------------------------------------------------
    # Continuous context
    #
    # PCA is fitted ONLY on internal training observations.
    # --------------------------------------------------------

    (
        context_predictor_scaler,
        pca,
        context_scaler,
        train_context
    ) = fit_context_transform(
        X_train
    )


    val_context = transform_context(
        X_val,
        context_predictor_scaler,
        pca,
        context_scaler
    )


    # --------------------------------------------------------
    # Target scaling
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
    # Tensors
    # --------------------------------------------------------

    train_x = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    train_context_tensor = torch.tensor(
        train_context,
        dtype=torch.float32,
        device=DEVICE
    )

    train_y = torch.tensor(
        y_train_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )


    val_x = torch.tensor(
        X_val_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    val_context_tensor = torch.tensor(
        val_context,
        dtype=torch.float32,
        device=DEVICE
    )

    val_y = torch.tensor(
        y_val_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ContinuousContextFiLM(
        input_dim=len(FEATURES),
        context_dim=CONTEXT_DIM
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
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()


        prediction = model(
            train_x,
            train_context_tensor
        )


        loss = criterion(
            prediction,
            train_y
        )

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_prediction = model(
                val_x,
                val_context_tensor
            )

            val_loss = criterion(
                val_prediction,
                val_y
            ).item()


        if val_loss < (
            best_loss - 1e-6
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
        feature_scaler,
        y_mean,
        y_std,
        context_predictor_scaler,
        pca,
        context_scaler
    )


# ============================================================
# OUTER SPATIAL CV
# ============================================================

outer_cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)


results = []
observations = []


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
    # STANDARD MLP
    # ========================================================

    torch.manual_seed(
        RANDOM_STATE
    )

    (
        standard_model,
        standard_scaler,
        standard_y_mean,
        standard_y_std
    ) = train_standard_mlp(
        X_outer_train,
        y_outer_train
    )


    X_test_standard_scaled = (
        standard_scaler.transform(
            X_test
        )
    )


    test_tensor = torch.tensor(
        X_test_standard_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    standard_model.eval()

    with torch.no_grad():

        standard_pred_scaled = (
            standard_model(
                test_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    standard_pred = (
        standard_pred_scaled
        * standard_y_std
        + standard_y_mean
    )


    # ========================================================
    # CONTINUOUS-CONTEXT FiLM
    # ========================================================

    torch.manual_seed(
        RANDOM_STATE
    )


    (
        film_model,
        film_scaler,
        film_y_mean,
        film_y_std,
        context_predictor_scaler,
        context_pca,
        context_scaler
    ) = train_film(
        X_outer_train,
        y_outer_train
    )


    X_test_film_scaled = (
        film_scaler.transform(
            X_test
        )
    )


    test_context = transform_context(
        X_test,
        context_predictor_scaler,
        context_pca,
        context_scaler
    )


    test_film_x = torch.tensor(
        X_test_film_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    test_context_tensor = torch.tensor(
        test_context,
        dtype=torch.float32,
        device=DEVICE
    )


    film_model.eval()

    with torch.no_grad():

        film_pred_scaled = (
            film_model(
                test_film_x,
                test_context_tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    film_pred = (
        film_pred_scaled
        * film_y_std
        + film_y_mean
    )


    # ========================================================
    # METRICS
    # ========================================================

    standard_r2 = r2_score(
        y_test,
        standard_pred
    )

    standard_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            standard_pred
        )
    )

    standard_mae = mean_absolute_error(
        y_test,
        standard_pred
    )


    film_r2 = r2_score(
        y_test,
        film_pred
    )

    film_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            film_pred
        )
    )

    film_mae = mean_absolute_error(
        y_test,
        film_pred
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("STANDARD MLP")

    print(
        "R²   =",
        round(
            standard_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            standard_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            standard_mae,
            6
        )
    )


    print()
    print("CONTINUOUS-CONTEXT FiLM")

    print(
        "R²   =",
        round(
            film_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            film_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            film_mae,
            6
        )
    )


    print()
    print("CHANGES")

    print(
        "ΔR²   =",
        round(
            film_r2 - standard_r2,
            6
        )
    )

    print(
        "ΔRMSE =",
        round(
            film_rmse - standard_rmse,
            6
        )
    )

    print(
        "ΔMAE  =",
        round(
            film_mae - standard_mae,
            6
        )
    )


    # ========================================================
    # PCA CONTEXT DIAGNOSTIC
    # ========================================================

    print()
    print(
        "PCA explained variance:",
        np.round(
            context_pca.explained_variance_ratio_,
            4
        )
    )


    # ========================================================
    # STORE FOLD RESULTS
    # ========================================================

    results.append({

        "Fold": fold,

        "Standard_MLP_R2":
            standard_r2,

        "Standard_MLP_RMSE":
            standard_rmse,

        "Standard_MLP_MAE":
            standard_mae,

        "Continuous_FiLM_R2":
            film_r2,

        "Continuous_FiLM_RMSE":
            film_rmse,

        "Continuous_FiLM_MAE":
            film_mae,

        "Delta_R2":
            film_r2 - standard_r2,

        "Delta_RMSE":
            film_rmse - standard_rmse,

        "Delta_MAE":
            film_mae - standard_mae
    })


    # ========================================================
    # STORE OBSERVATIONS
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


    meta["Standard_MLP_Predicted"] = (
        standard_pred
    )

    meta["Standard_MLP_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - standard_pred
        )
    )


    meta["Continuous_FiLM_Predicted"] = (
        film_pred
    )

    meta["Continuous_FiLM_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - film_pred
        )
    )


    for component in range(
        CONTEXT_DIM
    ):

        meta[
            f"Context_PC{component + 1}"
        ] = test_context[
            :,
            component
        ]


    observations.append(
        meta
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

observations_df = pd.concat(
    observations,
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Model": [
        "Standard_MLP",
        "Continuous_Context_FiLM"
    ],

    "Mean_R2": [
        results_df[
            "Standard_MLP_R2"
        ].mean(),

        results_df[
            "Continuous_FiLM_R2"
        ].mean()
    ],

    "SD_R2": [
        results_df[
            "Standard_MLP_R2"
        ].std(ddof=1),

        results_df[
            "Continuous_FiLM_R2"
        ].std(ddof=1)
    ],

    "Mean_RMSE": [
        results_df[
            "Standard_MLP_RMSE"
        ].mean(),

        results_df[
            "Continuous_FiLM_RMSE"
        ].mean()
    ],

    "SD_RMSE": [
        results_df[
            "Standard_MLP_RMSE"
        ].std(ddof=1),

        results_df[
            "Continuous_FiLM_RMSE"
        ].std(ddof=1)
    ],

    "Mean_MAE": [
        results_df[
            "Standard_MLP_MAE"
        ].mean(),

        results_df[
            "Continuous_FiLM_MAE"
        ].mean()
    ],

    "SD_MAE": [
        results_df[
            "Standard_MLP_MAE"
        ].std(ddof=1),

        results_df[
            "Continuous_FiLM_MAE"
        ].std(ddof=1)
    ]
})


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("============================================================")
print("FINAL CONTINUOUS-CONTEXT FiLM SUMMARY")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("FOLD-BY-FOLD CHANGES")
print("============================================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "YCEO_ContinuousFiLM_FoldResults.csv",
    index=False
)

summary.to_csv(
    "YCEO_ContinuousFiLM_Summary.csv",
    index=False
)

observations_df.to_csv(
    "YCEO_ContinuousFiLM_ObservationResults.csv",
    index=False
)


print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_ContinuousFiLM_FoldResults.csv"
)

print(
    "YCEO_ContinuousFiLM_Summary.csv"
)

print(
    "YCEO_ContinuousFiLM_ObservationResults.csv"
)

print()
print(
    "Continuous-context FiLM experiment complete."
)