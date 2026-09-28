import os
import random
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
N_OUTER_FOLDS = 5
INTERNAL_VAL_SIZE = 0.15

N_DOMAINS = 5

EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 64
EXPERT_HIDDEN_DIM = 32
DROPOUT = 0.10

PATIENCE = 20

DOMAIN_LOSS_WEIGHT = 0.10


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
    subset=FEATURES + [TARGET, "Lat", "Lon"]
).copy()

df = df[df["Population"] >= 0].copy()

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
print("LEAKAGE-FREE DOMAIN-CONDITIONED MIXTURE OF EXPERTS")
print("============================================================")

print("Dataset:", df.shape)
print("Features:", len(FEATURES))
print("Spatial blocks:", groups.nunique())
print("Domains:", N_DOMAINS)
print("Device:", DEVICE)


# ============================================================
# DOMAIN DISCOVERY
# ============================================================

def fit_domain_discovery(X_train):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_train)

    pca = PCA(
        n_components=min(5, X_scaled.shape[1]),
        random_state=RANDOM_STATE
    )

    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(
        n_clusters=N_DOMAINS,
        n_init=20,
        random_state=RANDOM_STATE
    )

    domains = kmeans.fit_predict(X_pca)

    return scaler, pca, kmeans, domains


def assign_domains(
    X_data,
    scaler,
    pca,
    kmeans
):

    X_scaled = scaler.transform(X_data)

    X_pca = pca.transform(X_scaled)

    return kmeans.predict(X_pca)


# ============================================================
# MIXTURE OF EXPERTS
# ============================================================

class DomainMoE(nn.Module):

    def __init__(
        self,
        input_dim,
        n_domains
    ):

        super().__init__()

        self.shared = nn.Sequential(

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

        self.gate = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                n_domains
            )
        )

        self.experts = nn.ModuleList(

            [
                nn.Sequential(

                    nn.Linear(
                        HIDDEN_DIM,
                        EXPERT_HIDDEN_DIM
                    ),

                    nn.ReLU(),

                    nn.Dropout(
                        DROPOUT
                    ),

                    nn.Linear(
                        EXPERT_HIDDEN_DIM,
                        1
                    )
                )

                for _ in range(n_domains)
            ]
        )

        self.domain_classifier = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_DIM,
                n_domains
            )
        )

    def forward(self, x):

        h = self.shared(x)

        gate_logits = self.gate(h)

        gate_weights = F.softmax(
            gate_logits,
            dim=1
        )

        expert_outputs = []

        for expert in self.experts:

            expert_outputs.append(
                expert(h)
            )

        expert_outputs = torch.cat(
            expert_outputs,
            dim=1
        )

        prediction = (
            gate_weights * expert_outputs
        ).sum(dim=1)

        domain_logits = self.domain_classifier(h)

        return (
            prediction,
            gate_weights,
            domain_logits
        )


# ============================================================
# TRAIN DOMAIN MoE
# ============================================================

def train_moe(
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

    inner_train_idx, inner_val_idx = next(
        splitter.split(
            X_outer_train,
            y_outer_train,
            groups=inner_groups
        )
    )

    X_train_raw = X_outer_train.iloc[
        inner_train_idx
    ][FEATURES]

    X_val_raw = X_outer_train.iloc[
        inner_val_idx
    ][FEATURES]

    y_train = y_outer_train.iloc[
        inner_train_idx
    ]

    y_val = y_outer_train.iloc[
        inner_val_idx
    ]


    # --------------------------------------------------------
    # DOMAIN DISCOVERY
    # --------------------------------------------------------

    (
        domain_scaler,
        domain_pca,
        domain_kmeans,
        train_domains
    ) = fit_domain_discovery(
        X_train_raw
    )

    val_domains = assign_domains(
        X_val_raw,
        domain_scaler,
        domain_pca,
        domain_kmeans
    )


    # --------------------------------------------------------
    # FEATURE SCALING
    # --------------------------------------------------------

    feature_scaler = StandardScaler()

    X_train_scaled = feature_scaler.fit_transform(
        X_train_raw
    )

    X_val_scaled = feature_scaler.transform(
        X_val_raw
    )


    # --------------------------------------------------------
    # TARGET SCALING
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
    # TENSORS
    # --------------------------------------------------------

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

    train_domains_tensor = torch.tensor(
        train_domains,
        dtype=torch.long,
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
    # MODEL
    # --------------------------------------------------------

    model = DomainMoE(
        input_dim=len(FEATURES),
        n_domains=N_DOMAINS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    prediction_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    best_val_loss = np.inf
    best_state = None
    patience_counter = 0


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        optimizer.zero_grad()

        (
            prediction,
            gate_weights,
            domain_logits
        ) = model(train_x)


        prediction_loss = prediction_criterion(
            prediction,
            train_y
        )

        domain_loss = domain_criterion(
            domain_logits,
            train_domains_tensor
        )

        total_loss = (
            prediction_loss
            +
            DOMAIN_LOSS_WEIGHT
            * domain_loss
        )

        total_loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # INTERNAL VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_prediction, _, _ = model(
                val_x
            )

            val_loss = prediction_criterion(
                val_prediction,
                val_y
            ).item()


        if val_loss < (
            best_val_loss - 1e-6
        ):

            best_val_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= PATIENCE:
            break


    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    return (
        model,
        feature_scaler,
        y_mean,
        y_std,
        domain_scaler,
        domain_pca,
        domain_kmeans
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
moe_results = []
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
    # BASELINE XGBOOST
    # ========================================================

    xgb = XGBRegressor(

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

    xgb.fit(
        X_outer_train[FEATURES],
        y_outer_train
    )

    xgb_pred = xgb.predict(
        X_test
    )


    # ========================================================
    # DOMAIN MoE
    # ========================================================

    (
        moe,
        feature_scaler,
        moe_y_mean,
        moe_y_std,
        domain_scaler,
        domain_pca,
        domain_kmeans
    ) = train_moe(
        X_outer_train,
        y_outer_train
    )


    X_test_scaled = feature_scaler.transform(
        X_test
    )

    test_tensor = torch.tensor(
        X_test_scaled,
        dtype=torch.float32,
        device=DEVICE
    )


    moe.eval()

    with torch.no_grad():

        (
            moe_pred_scaled,
            gate_weights,
            domain_logits
        ) = moe(
            test_tensor
        )


    moe_pred = (
        moe_pred_scaled
        .detach()
        .cpu()
        .numpy()
        * moe_y_std
        + moe_y_mean
    )


    gate_weights_np = (
        gate_weights
        .detach()
        .cpu()
        .numpy()
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


    moe_r2 = r2_score(
        y_test,
        moe_pred
    )

    moe_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            moe_pred
        )
    )

    moe_mae = mean_absolute_error(
        y_test,
        moe_pred
    )


    print()
    print("BASELINE XGBOOST")

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
    print("DOMAIN-CONDITIONED MoE")

    print(
        "R²   =",
        round(moe_r2, 6)
    )

    print(
        "RMSE =",
        round(moe_rmse, 6)
    )

    print(
        "MAE  =",
        round(moe_mae, 6)
    )


    print()
    print("CHANGES")

    print(
        "ΔR²   =",
        round(
            moe_r2 - xgb_r2,
            6
        )
    )

    print(
        "ΔRMSE =",
        round(
            moe_rmse - xgb_rmse,
            6
        )
    )

    print(
        "ΔMAE  =",
        round(
            moe_mae - xgb_mae,
            6
        )
    )


    # ========================================================
    # GATE DIAGNOSTICS
    # ========================================================

    gate_entropy = (
        -np.sum(
            gate_weights_np
            *
            np.log(
                gate_weights_np + 1e-12
            ),
            axis=1
        ).mean()
    )


    primary_expert = np.argmax(
        gate_weights_np,
        axis=1
    )


    gate_usage = (
        pd.Series(
            primary_expert
        )
        .value_counts(
            normalize=True
        )
        .sort_index()
    )


    print()
    print(
        "Mean gate entropy:",
        round(
            gate_entropy,
            6
        )
    )

    print()
    print("Gate usage:")

    print(
        gate_usage.to_string()
    )


    # ========================================================
    # STORE
    # ========================================================

    xgb_results.append({

        "Fold": fold,

        "R2": xgb_r2,

        "RMSE": xgb_rmse,

        "MAE": xgb_mae
    })


    moe_results.append({

        "Fold": fold,

        "R2": moe_r2,

        "RMSE": moe_rmse,

        "MAE": moe_mae,

        "Gate_Entropy":
            gate_entropy
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

    meta["XGBoost_Absolute_Error"] = np.abs(
        y_test.to_numpy()
        - xgb_pred
    )

    meta["MoE_Predicted"] = (
        moe_pred
    )

    meta["MoE_Absolute_Error"] = np.abs(
        y_test.to_numpy()
        - moe_pred
    )

    meta["MoE_Primary_Expert"] = (
        primary_expert
    )


    for domain_id in range(
        N_DOMAINS
    ):

        meta[
            f"Gate_Weight_{domain_id}"
        ] = gate_weights_np[
            :,
            domain_id
        ]


    observation_results.append(
        meta
    )


# ============================================================
# DATAFRAMES
# ============================================================

xgb_df = pd.DataFrame(
    xgb_results
)

moe_df = pd.DataFrame(
    moe_results
)

obs_df = pd.concat(
    observation_results,
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Model": [
        "XGBoost",
        "Domain_Conditioned_MoE"
    ],

    "Mean_R2": [
        xgb_df["R2"].mean(),
        moe_df["R2"].mean()
    ],

    "SD_R2": [
        xgb_df["R2"].std(
            ddof=1
        ),
        moe_df["R2"].std(
            ddof=1
        )
    ],

    "Mean_RMSE": [
        xgb_df["RMSE"].mean(),
        moe_df["RMSE"].mean()
    ],

    "SD_RMSE": [
        xgb_df["RMSE"].std(
            ddof=1
        ),
        moe_df["RMSE"].std(
            ddof=1
        )
    ],

    "Mean_MAE": [
        xgb_df["MAE"].mean(),
        moe_df["MAE"].mean()
    ],

    "SD_MAE": [
        xgb_df["MAE"].std(
            ddof=1
        ),
        moe_df["MAE"].std(
            ddof=1
        )
    ]
})


# ============================================================
# PAIRED DIFFERENCES
# ============================================================

paired = xgb_df.merge(
    moe_df,
    on="Fold",
    suffixes=(
        "_XGB",
        "_MoE"
    )
)


paired["Delta_R2"] = (
    paired["R2_MoE"]
    - paired["R2_XGB"]
)

paired["Delta_RMSE"] = (
    paired["RMSE_MoE"]
    - paired["RMSE_XGB"]
)

paired["Delta_MAE"] = (
    paired["MAE_MoE"]
    - paired["MAE_XGB"]
)


# ============================================================
# PRINT FINAL
# ============================================================

print()
print("============================================================")
print("FINAL MoE SUMMARY")
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
    paired[
        [
            "Fold",
            "Delta_R2",
            "Delta_RMSE",
            "Delta_MAE"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

xgb_df.to_csv(
    "YCEO_DomainMoE_XGBoost_FoldResults.csv",
    index=False
)

moe_df.to_csv(
    "YCEO_DomainMoE_FoldResults.csv",
    index=False
)

summary.to_csv(
    "YCEO_DomainMoE_Summary.csv",
    index=False
)

paired.to_csv(
    "YCEO_DomainMoE_Paired_Differences.csv",
    index=False
)

obs_df.to_csv(
    "YCEO_DomainMoE_Observation_Results.csv",
    index=False
)


print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_DomainMoE_XGBoost_FoldResults.csv"
)

print(
    "YCEO_DomainMoE_FoldResults.csv"
)

print(
    "YCEO_DomainMoE_Summary.csv"
)

print(
    "YCEO_DomainMoE_Paired_Differences.csv"
)

print(
    "YCEO_DomainMoE_Observation_Results.csv"
)

print()
print(
    "Domain-conditioned MoE experiment complete."
)