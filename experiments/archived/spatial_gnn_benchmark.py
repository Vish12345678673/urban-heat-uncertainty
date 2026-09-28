import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import torch
import torch.nn as nn
import torch.nn.functional as F


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

HIDDEN_DIM = 64

DROPOUT = 0.10

PATIENCE = 20

K_NEIGHBORS = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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

os.environ["PYTHONHASHSEED"] = str(
    RANDOM_STATE
)

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


X = df[FEATURES].copy()

y = df[TARGET].copy()

groups = df["spatial_block"].copy()


print()
print("================================================")
print("LEAKAGE-FREE SPATIAL GRAPH BENCHMARK")
print("================================================")

print(
    "Dataset:",
    df.shape
)

print(
    "Spatial blocks:",
    groups.nunique()
)

print(
    "Graph neighbors:",
    K_NEIGHBORS
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# STANDARD MLP
# ============================================================

class MLPBaseline(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

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
        x
    ):

        return self.net(
            x
        ).squeeze(-1)


# ============================================================
# SIMPLE GRAPH CONVOLUTION
# ============================================================

class GraphConvLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim
    ):

        super().__init__()

        self.linear_self = nn.Linear(
            in_dim,
            out_dim
        )

        self.linear_neighbor = nn.Linear(
            in_dim,
            out_dim
        )


    def forward(
        self,
        x,
        adjacency
    ):

        neighbor_message = (
            adjacency @ x
        )

        return (
            self.linear_self(x)
            +
            self.linear_neighbor(
                neighbor_message
            )
        )


# ============================================================
# GRAPH MODEL
# ============================================================

class SpatialGNN(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.conv1 = GraphConvLayer(
            input_dim,
            HIDDEN_DIM
        )

        self.conv2 = GraphConvLayer(
            HIDDEN_DIM,
            HIDDEN_DIM
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.output = nn.Linear(
            HIDDEN_DIM,
            1
        )


    def forward(
        self,
        x,
        adjacency
    ):

        h = self.conv1(
            x,
            adjacency
        )

        h = F.relu(
            h
        )

        h = self.dropout(
            h
        )

        h = self.conv2(
            h,
            adjacency
        )

        h = F.relu(
            h
        )

        h = self.dropout(
            h
        )

        return self.output(
            h
        ).squeeze(-1)


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_matrix(
    lat,
    lon
):

    lat = np.radians(
        lat
    )

    lon = np.radians(
        lon
    )

    dlat = (
        lat[:, None]
        - lat[None, :]
    )

    dlon = (
        lon[:, None]
        - lon[None, :]
    )

    a = (
        np.sin(
            dlat / 2
        ) ** 2
        +
        np.cos(lat[:, None])
        *
        np.cos(lat[None, :])
        *
        np.sin(
            dlon / 2
        ) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(
            np.clip(
                a,
                0,
                1
            )
        )
    )

    earth_radius_km = 6371.0

    return (
        earth_radius_km
        * c
    )


# ============================================================
# KNN GRAPH
# ============================================================

def build_knn_adjacency(
    lat,
    lon,
    k
):

    n = len(lat)

    distances = haversine_matrix(
        np.asarray(lat),
        np.asarray(lon)
    )

    # Never connect a node to itself.
    np.fill_diagonal(
        distances,
        np.inf
    )

    adjacency = np.zeros(
        (
            n,
            n
        ),
        dtype=np.float32
    )


    for i in range(n):

        kk = min(
            k,
            n - 1
        )

        nearest = np.argpartition(
            distances[i],
            kk
        )[:kk]

        adjacency[
            i,
            nearest
        ] = 1.0


    # Symmetrize graph.
    adjacency = np.maximum(
        adjacency,
        adjacency.T
    )


    # Add self-loops.
    adjacency += np.eye(
        n,
        dtype=np.float32
    )


    # Normalize adjacency:
    #
    # D^(-1/2) A D^(-1/2)
    #

    degree = adjacency.sum(
        axis=1
    )

    inv_sqrt = 1.0 / np.sqrt(
        degree
        + 1e-12
    )

    normalized = (
        inv_sqrt[:, None]
        *
        adjacency
        *
        inv_sqrt[None, :]
    )


    return normalized


# ============================================================
# INTERNAL TRAINING
# ============================================================

def train_mlp_internal(
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

    train_target = torch.tensor(
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


    model = MLPBaseline(
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


    best_loss = np.inf
    best_state = None
    patience = 0


    for epoch in range(
        EPOCHS
    ):

        model.train()

        optimizer.zero_grad()

        pred = model(
            train_tensor
        )

        loss = criterion(
            pred,
            train_target
        )

        loss.backward()

        optimizer.step()


        model.eval()

        with torch.no_grad():

            val_pred = model(
                val_tensor
            )

            val_loss = criterion(
                val_pred,
                val_target
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

            patience = 0

        else:

            patience += 1


        if patience >= PATIENCE:

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
# TRAIN GRAPH MODEL
# ============================================================

def train_graph_model(
    X_outer_train,
    y_outer_train
):

    # --------------------------------------------------------
    # INTERNAL SPATIAL VALIDATION
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
    # Coordinates
    # --------------------------------------------------------

    lat_train = X_outer_train.iloc[
        train_idx
    ]["_Lat_"].to_numpy()

    lon_train = X_outer_train.iloc[
        train_idx
    ]["_Lon_"].to_numpy()

    lat_val = X_outer_train.iloc[
        val_idx
    ]["_Lat_"].to_numpy()

    lon_val = X_outer_train.iloc[
        val_idx
    ]["_Lon_"].to_numpy()


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
    # Internal TRAIN graph only.
    #
    # Important:
    # The graph used for optimization is constructed from
    # training nodes only.
    # --------------------------------------------------------

    train_adj = build_knn_adjacency(
        lat_train,
        lon_train,
        K_NEIGHBORS
    )


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

    train_a = torch.tensor(
        train_adj,
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


    # --------------------------------------------------------
    # For internal validation we construct a validation graph
    # from validation nodes only.
    #
    # There are NO edges between training and validation nodes.
    # --------------------------------------------------------

    val_adj = build_knn_adjacency(
        lat_val,
        lon_val,
        K_NEIGHBORS
    )


    val_a = torch.tensor(
        val_adj,
        dtype=torch.float32,
        device=DEVICE
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SpatialGNN(
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


    best_loss = np.inf
    best_state = None
    patience = 0


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        model.train()

        optimizer.zero_grad()

        pred = model(
            train_x,
            train_a
        )

        loss = criterion(
            pred,
            train_y
        )

        loss.backward()

        optimizer.step()


        model.eval()

        with torch.no_grad():

            val_pred = model(
                val_x,
                val_a
            )

            val_loss = criterion(
                val_pred,
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

            patience = 0

        else:

            patience += 1


        if patience >= PATIENCE:

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


baseline_results = []
gnn_results = []
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
    print(
        "OUTER SPATIAL FOLD",
        fold
    )
    print("================================================")


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
    # MLP BASELINE
    # ========================================================

    (
        mlp,
        mlp_scaler,
        mlp_y_mean,
        mlp_y_std
    ) = train_mlp_internal(
        X_outer_train,
        y_outer_train
    )


    X_test_scaled = mlp_scaler.transform(
        X_test
    )

    test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    mlp.eval()

    with torch.no_grad():

        mlp_pred_scaled = mlp(
            test_tensor
        ).detach().cpu().numpy()


    mlp_pred = (
        mlp_pred_scaled
        * mlp_y_std
        + mlp_y_mean
    )


    # ========================================================
    # GNN MODEL
    # ========================================================
    #
    # The GNN is trained using ONLY outer training nodes.
    #
    # The outer test nodes are NOT used during training.
    #
    # For the test prediction graph, test nodes form a graph
    # ONLY among themselves. This is deliberately conservative.
    #
    # Therefore the graph model does not get information from
    # test labels or test-training edges.
    # ========================================================

    (
        gnn,
        gnn_scaler,
        gnn_y_mean,
        gnn_y_std
    ) = train_graph_model(
        X_outer_train,
        y_outer_train
    )


    X_test_scaled_gnn = gnn_scaler.transform(
        X_test
    )


    test_lat = df.iloc[
        test_idx
    ]["Lat"].to_numpy()

    test_lon = df.iloc[
        test_idx
    ]["Lon"].to_numpy()


    test_adj = build_knn_adjacency(
        test_lat,
        test_lon,
        K_NEIGHBORS
    )


    test_x_gnn = torch.tensor(
        X_test_scaled_gnn,
        dtype=torch.float32,
        device=DEVICE
    )

    test_a_gnn = torch.tensor(
        test_adj,
        dtype=torch.float32,
        device=DEVICE
    )


    gnn.eval()

    with torch.no_grad():

        gnn_pred_scaled = gnn(
            test_x_gnn,
            test_a_gnn
        ).detach().cpu().numpy()


    gnn_pred = (
        gnn_pred_scaled
        * gnn_y_std
        + gnn_y_mean
    )


    # ========================================================
    # METRICS
    # ========================================================

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


    gnn_r2 = r2_score(
        y_test,
        gnn_pred
    )

    gnn_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            gnn_pred
        )
    )

    gnn_mae = mean_absolute_error(
        y_test,
        gnn_pred
    )


    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("STANDARD MLP")

    print(
        "R²   =",
        round(
            mlp_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            mlp_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            mlp_mae,
            6
        )
    )


    print()
    print("SPATIAL GNN")

    print(
        "R²   =",
        round(
            gnn_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            gnn_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            gnn_mae,
            6
        )
    )


    # ========================================================
    # STORE
    # ========================================================

    baseline_results.append({

        "Fold": fold,

        "Model": "Standard_MLP",

        "R2": mlp_r2,

        "RMSE": mlp_rmse,

        "MAE": mlp_mae
    })


    gnn_results.append({

        "Fold": fold,

        "Model": "Spatial_GNN",

        "R2": gnn_r2,

        "RMSE": gnn_rmse,

        "MAE": gnn_mae
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

    meta["MLP_Predicted"] = (
        mlp_pred
    )

    meta["MLP_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - mlp_pred
        )
    )

    meta["GNN_Predicted"] = (
        gnn_pred
    )

    meta["GNN_Absolute_Error"] = (
        np.abs(
            y_test.to_numpy()
            - gnn_pred
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

gnn_df = pd.DataFrame(
    gnn_results
)

comparison_df = pd.concat(
    [
        baseline_df,
        gnn_df
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
        )
    )
    .reset_index()
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("================================================")
print("FINAL MODEL SUMMARY")
print("================================================")

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

comparison_df.to_csv(
    "YCEO_SpatialGNN_Model_Comparison.csv",
    index=False
)

summary.to_csv(
    "YCEO_SpatialGNN_Model_Summary.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_SpatialGNN_Observation_Results.csv",
    index=False
)


print()
print("================================================")
print("FILES CREATED")
print("================================================")

print(
    "YCEO_SpatialGNN_Model_Comparison.csv"
)

print(
    "YCEO_SpatialGNN_Model_Summary.csv"
)

print(
    "YCEO_SpatialGNN_Observation_Results.csv"
)

print()
print(
    "Spatial GNN benchmark complete."
)