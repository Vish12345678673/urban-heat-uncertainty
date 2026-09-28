import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit
)
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


# ============================================================
# CONFIGURATION
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

DOMAIN_EMBED_DIM = 16

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
    (df["NDVI"] >= -1)
    &
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


X_with_coords = X.copy()

X_with_coords["_Lat_"] = df["Lat"].values

X_with_coords["_Lon_"] = df["Lon"].values


print()
print("============================================================")
print("DOMAIN-CONDITIONED MLP ABLATION")
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
    "Domains:",
    N_DOMAINS
)

print(
    "Device:",
    DEVICE
)


# ============================================================
# DOMAIN DISCOVERY
# ============================================================

def fit_domains(
    X_train
):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train[FEATURES]
    )


    pca = PCA(
        n_components=min(
            5,
            len(FEATURES)
        ),
        random_state=RANDOM_STATE
    )

    X_pca = pca.fit_transform(
        X_scaled
    )


    kmeans = KMeans(
        n_clusters=N_DOMAINS,
        n_init=20,
        random_state=RANDOM_STATE
    )

    labels = kmeans.fit_predict(
        X_pca
    )


    return (
        scaler,
        pca,
        kmeans,
        labels
    )


def assign_domains(
    X_data,
    scaler,
    pca,
    kmeans
):

    X_scaled = scaler.transform(
        X_data[FEATURES]
    )

    X_pca = pca.transform(
        X_scaled
    )

    return kmeans.predict(
        X_pca
    )


# ============================================================
# COMMON MLP BLOCK
# ============================================================

class SharedEncoder(nn.Module):

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

            nn.BatchNorm1d(
                HIDDEN_DIM
            ),

            nn.Dropout(
                DROPOUT
            )
        )


    def forward(
        self,
        x
    ):

        return self.net(
            x
        )


# ============================================================
# MODEL A: STANDARD MLP
# ============================================================

class StandardMLP(nn.Module):

    def __init__(
        self,
        input_dim
    ):

        super().__init__()

        self.encoder = SharedEncoder(
            input_dim
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


    def forward(
        self,
        x,
        domain=None
    ):

        h = self.encoder(
            x
        )

        return self.head(
            h
        ).squeeze(-1)


# ============================================================
# MODEL B: ONE-HOT DOMAIN CONDITIONING
# ============================================================

class OneHotDomainMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        n_domains
    ):

        super().__init__()

        self.encoder = SharedEncoder(
            input_dim
        )

        self.head = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM + n_domains,
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
        domain
    ):

        h = self.encoder(
            x
        )

        domain_onehot = torch.zeros(
            (
                x.shape[0],
                N_DOMAINS
            ),
            device=x.device
        )

        domain_onehot.scatter_(
            1,
            domain.unsqueeze(1),
            1.0
        )

        combined = torch.cat(
            [
                h,
                domain_onehot
            ],
            dim=1
        )

        return self.head(
            combined
        ).squeeze(-1)


# ============================================================
# MODEL C: LEARNED DOMAIN EMBEDDING
# ============================================================

class DomainEmbeddingMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        n_domains
    ):

        super().__init__()

        self.encoder = SharedEncoder(
            input_dim
        )

        self.embedding = nn.Embedding(
            n_domains,
            DOMAIN_EMBED_DIM
        )

        self.head = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM + DOMAIN_EMBED_DIM,
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
        domain
    ):

        h = self.encoder(
            x
        )

        d = self.embedding(
            domain
        )

        combined = torch.cat(
            [
                h,
                d
            ],
            dim=1
        )

        return self.head(
            combined
        ).squeeze(-1)


# ============================================================
# MODEL D: FiLM DOMAIN CONDITIONING
# ============================================================

class FiLMDomainMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        n_domains
    ):

        super().__init__()

        self.encoder = SharedEncoder(
            input_dim
        )

        self.embedding = nn.Embedding(
            n_domains,
            DOMAIN_EMBED_DIM
        )

        self.gamma = nn.Linear(
            DOMAIN_EMBED_DIM,
            HIDDEN_DIM
        )

        self.beta = nn.Linear(
            DOMAIN_EMBED_DIM,
            HIDDEN_DIM
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


    def forward(
        self,
        x,
        domain
    ):

        h = self.encoder(
            x
        )

        d = self.embedding(
            domain
        )

        gamma = self.gamma(
            d
        )

        beta = self.beta(
            d
        )

        # Feature-wise Linear Modulation.
        h = (
            (1.0 + gamma)
            * h
            + beta
        )

        h = torch.relu(
            h
        )

        return self.head(
            h
        ).squeeze(-1)


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    domain_train,
    X_val,
    y_val,
    domain_val
):

    # --------------------------------------------------------
    # Feature scaling
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

    train_y = torch.tensor(
        y_train_scaled.to_numpy(),
        dtype=torch.float32,
        device=DEVICE
    )

    train_domain_tensor = torch.tensor(
        domain_train,
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

    val_domain_tensor = torch.tensor(
        domain_val,
        dtype=torch.long,
        device=DEVICE
    )


    model = model.to(
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
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        model.train()

        optimizer.zero_grad()


        # Standard MLP does not require domain.
        if isinstance(
            model,
            StandardMLP
        ):

            prediction = model(
                train_x
            )

        else:

            prediction = model(
                train_x,
                train_domain_tensor
            )


        loss = criterion(
            prediction,
            train_y
        )

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # INTERNAL VALIDATION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            if isinstance(
                model,
                StandardMLP
            ):

                val_prediction = model(
                    val_x
                )

            else:

                val_prediction = model(
                    val_x,
                    val_domain_tensor
                )


            val_loss = criterion(
                val_prediction,
                val_y
            ).item()


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

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


    # --------------------------------------------------------
    # RESTORE
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
# MODEL FACTORIES
# ============================================================

def make_models():

    return {

        "Standard_MLP":
            StandardMLP(
                len(FEATURES)
            ),

        "OneHot_Domain_MLP":
            OneHotDomainMLP(
                len(FEATURES),
                N_DOMAINS
            ),

        "Embedding_Domain_MLP":
            DomainEmbeddingMLP(
                len(FEATURES),
                N_DOMAINS
            ),

        "FiLM_Domain_MLP":
            FiLMDomainMLP(
                len(FEATURES),
                N_DOMAINS
            )
    }


# ============================================================
# OUTER SPATIAL CV
# ============================================================

outer_cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)

all_fold_results = []

all_observations = []


for fold, (
    outer_train_idx,
    outer_test_idx
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


    X_outer = X_with_coords.iloc[
        outer_train_idx
    ].copy()

    y_outer = y.iloc[
        outer_train_idx
    ].copy()


    X_test = X.iloc[
        outer_test_idx
    ].copy()

    y_test = y.iloc[
        outer_test_idx
    ].copy()


    # ========================================================
    # INTERNAL SPATIAL VALIDATION
    # ========================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=INTERNAL_VAL_SIZE,
        random_state=RANDOM_STATE
    )


    inner_lat = np.floor(
        X_outer["_Lat_"] / 10
    ).astype(int)

    inner_lon = np.floor(
        X_outer["_Lon_"] / 10
    ).astype(int)


    inner_groups = (
        inner_lat.astype(str)
        + "_"
        + inner_lon.astype(str)
    )


    inner_train_idx, inner_val_idx = next(

        splitter.split(
            X_outer,
            y_outer,
            groups=inner_groups
        )
    )


    X_inner_train = X_outer.iloc[
        inner_train_idx
    ]

    X_inner_val = X_outer.iloc[
        inner_val_idx
    ]

    y_inner_train = y_outer.iloc[
        inner_train_idx
    ]

    y_inner_val = y_outer.iloc[
        inner_val_idx
    ]


    # ========================================================
    # DOMAIN DISCOVERY USING INTERNAL TRAIN ONLY
    # ========================================================

    (
        domain_scaler,
        domain_pca,
        domain_kmeans,
        train_domains
    ) = fit_domains(
        X_inner_train
    )


    val_domains = assign_domains(
        X_inner_val,
        domain_scaler,
        domain_pca,
        domain_kmeans
    )


    test_domains = assign_domains(
        X_test,
        domain_scaler,
        domain_pca,
        domain_kmeans
    )


    print()
    print("Internal training domain counts:")

    print(
        pd.Series(
            train_domains
        )
        .value_counts()
        .sort_index()
        .to_string()
    )


    print()
    print("Outer test domain counts:")

    print(
        pd.Series(
            test_domains
        )
        .value_counts()
        .sort_index()
        .to_string()
    )


    # ========================================================
    # TRAIN FOUR MODELS
    # ========================================================

    models = make_models()


    for model_name, model in models.items():

        print()
        print(
            "--------------------------------------------"
        )

        print(
            model_name
        )

        print(
            "--------------------------------------------"
        )


        # ----------------------------------------------------
        # Reset seed for comparable initialization.
        # ----------------------------------------------------

        torch.manual_seed(
            RANDOM_STATE
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                RANDOM_STATE
            )


        (
            trained_model,
            feature_scaler,
            y_mean,
            y_std
        ) = train_model(

            model,

            X_inner_train[FEATURES],

            y_inner_train,

            train_domains,

            X_inner_val[FEATURES],

            y_inner_val,

            val_domains
        )


        # ----------------------------------------------------
        # OUTER TEST
        # ----------------------------------------------------

        X_test_scaled = feature_scaler.transform(
            X_test
        )


        test_x = torch.tensor(
            X_test_scaled,
            dtype=torch.float32,
            device=DEVICE
        )


        test_domain_tensor = torch.tensor(
            test_domains,
            dtype=torch.long,
            device=DEVICE
        )


        trained_model.eval()


        with torch.no_grad():

            if isinstance(
                trained_model,
                StandardMLP
            ):

                pred_scaled = trained_model(
                    test_x
                )

            else:

                pred_scaled = trained_model(
                    test_x,
                    test_domain_tensor
                )


        prediction = (
            pred_scaled
            .detach()
            .cpu()
            .numpy()
            * y_std
            + y_mean
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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


        all_fold_results.append({

            "Fold":
                fold,

            "Model":
                model_name,

            "R2":
                r2,

            "RMSE":
                rmse,

            "MAE":
                mae
        })


        # ----------------------------------------------------
        # OBSERVATIONS
        # ----------------------------------------------------

        meta = df.iloc[
            outer_test_idx
        ][
            [
                "Code",
                "Lat",
                "Lon",
                "spatial_block"
            ]
        ].copy()


        meta["Fold"] = fold

        meta["Model"] = model_name

        meta["Domain"] = (
            test_domains
        )

        meta["Observed"] = (
            y_test.to_numpy()
        )

        meta["Predicted"] = (
            prediction
        )

        meta["Absolute_Error"] = (
            np.abs(
                y_test.to_numpy()
                - prediction
            )
        )


        all_observations.append(
            meta
        )


# ============================================================
# RESULTS
# ============================================================

fold_df = pd.DataFrame(
    all_fold_results
)

observation_df = pd.concat(
    all_observations,
    ignore_index=True
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    fold_df
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
# DOMAIN-WISE TEST PERFORMANCE
# ============================================================

domain_performance = (
    observation_df
    .groupby(
        [
            "Model",
            "Domain"
        ]
    )
    .agg(

        Observations=(
            "Code",
            "count"
        ),

        MAE=(
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
# PAIRED FOLD COMPARISON
# ============================================================

paired_rows = []


for fold in sorted(
    fold_df["Fold"].unique()
):

    temp = fold_df[
        fold_df["Fold"] == fold
    ]


    baseline = temp[
        temp["Model"]
        ==
        "Standard_MLP"
    ].iloc[0]


    for model_name in [
        "OneHot_Domain_MLP",
        "Embedding_Domain_MLP",
        "FiLM_Domain_MLP"
    ]:

        candidate = temp[
            temp["Model"]
            ==
            model_name
        ].iloc[0]


        paired_rows.append({

            "Fold":
                fold,

            "Model":
                model_name,

            "Delta_R2":
                candidate["R2"]
                - baseline["R2"],

            "Delta_RMSE":
                candidate["RMSE"]
                - baseline["RMSE"],

            "Delta_MAE":
                candidate["MAE"]
                - baseline["MAE"]
        })


paired_df = pd.DataFrame(
    paired_rows
)


# ============================================================
# PRINT FINAL
# ============================================================

print()
print("============================================================")
print("FINAL ABLATION SUMMARY")
print("============================================================")

print(
    summary.to_string(
        index=False
    )
)


print()
print("============================================================")
print("PAIRED IMPROVEMENTS VS STANDARD MLP")
print("============================================================")

print(
    paired_df.to_string(
        index=False
    )
)


print()
print("============================================================")
print("DOMAIN-WISE TEST PERFORMANCE")
print("============================================================")

print(
    domain_performance.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

fold_df.to_csv(
    "YCEO_DomainConditionedMLP_FoldResults.csv",
    index=False
)

summary.to_csv(
    "YCEO_DomainConditionedMLP_Summary.csv",
    index=False
)

paired_df.to_csv(
    "YCEO_DomainConditionedMLP_PairedDifferences.csv",
    index=False
)

domain_performance.to_csv(
    "YCEO_DomainConditionedMLP_DomainPerformance.csv",
    index=False
)

observation_df.to_csv(
    "YCEO_DomainConditionedMLP_ObservationResults.csv",
    index=False
)


print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_DomainConditionedMLP_FoldResults.csv"
)

print(
    "YCEO_DomainConditionedMLP_Summary.csv"
)

print(
    "YCEO_DomainConditionedMLP_PairedDifferences.csv"
)

print(
    "YCEO_DomainConditionedMLP_DomainPerformance.csv"
)

print(
    "YCEO_DomainConditionedMLP_ObservationResults.csv"
)

print()
print(
    "Domain-conditioned MLP ablation complete."
)