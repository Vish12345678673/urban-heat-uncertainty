import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42

N_OUTER_FOLDS = 5

N_DOMAINS = 5

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


print()
print("============================================================")
print("PREDICTOR-SPACE DOMAIN GENERALIZATION")
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
    "Environmental domains:",
    N_DOMAINS
)


# ============================================================
# XGBOOST
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
# DOMAIN CREATION
# ============================================================

def fit_domain_model(
    X_train
):

    # --------------------------------------------------------
    # Scale using training data only.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )


    # --------------------------------------------------------
    # Reduce dimensionality.
    #
    # PCA itself is fitted only on training observations.
    # --------------------------------------------------------

    n_components = min(
        5,
        X_scaled.shape[1]
    )

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    X_pca = pca.fit_transform(
        X_scaled
    )


    # --------------------------------------------------------
    # K-means domains.
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=N_DOMAINS,
        random_state=RANDOM_STATE,
        n_init=20
    )

    train_domains = kmeans.fit_predict(
        X_pca
    )


    return (
        scaler,
        pca,
        kmeans,
        train_domains
    )


# ============================================================
# ASSIGN NEW OBSERVATIONS TO TRAINING DOMAINS
# ============================================================

def assign_domains(
    X_data,
    scaler,
    pca,
    kmeans
):

    X_scaled = scaler.transform(
        X_data
    )

    X_pca = pca.transform(
        X_scaled
    )

    domains = kmeans.predict(
        X_pca
    )

    return domains


# ============================================================
# OUTER SPATIAL CV
# ============================================================

cv = GroupKFold(
    n_splits=N_OUTER_FOLDS
)


erm_results = []

balanced_results = []

domain_results = []


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
    print("============================================================")
    print(
        "OUTER SPATIAL FOLD",
        fold
    )
    print("============================================================")


    X_train = X.iloc[
        train_idx
    ].copy()

    y_train = y.iloc[
        train_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()

    y_test = y.iloc[
        test_idx
    ].copy()


    # ========================================================
    # FIT PREDICTOR-SPACE DOMAINS ON TRAINING DATA ONLY
    # ========================================================

    (
        domain_scaler,
        domain_pca,
        domain_kmeans,
        train_domains
    ) = fit_domain_model(
        X_train
    )


    test_domains = assign_domains(
        X_test,
        domain_scaler,
        domain_pca,
        domain_kmeans
    )


    # ========================================================
    # PRINT DOMAIN DISTRIBUTIONS
    # ========================================================

    print()
    print("Training domain counts:")

    train_domain_counts = (
        pd.Series(
            train_domains
        )
        .value_counts()
        .sort_index()
    )

    print(
        train_domain_counts.to_string()
    )


    print()
    print("Outer-test domain counts:")

    test_domain_counts = (
        pd.Series(
            test_domains
        )
        .value_counts()
        .sort_index()
    )

    print(
        test_domain_counts.to_string()
    )


    # ========================================================
    # STANDARD ERM XGBOOST
    # ========================================================

    erm = make_model()

    erm.fit(
        X_train,
        y_train
    )

    erm_pred = erm.predict(
        X_test
    )


    erm_r2 = r2_score(
        y_test,
        erm_pred
    )

    erm_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            erm_pred
        )
    )

    erm_mae = mean_absolute_error(
        y_test,
        erm_pred
    )


    # ========================================================
    # DOMAIN-BALANCED XGBOOST
    # ========================================================
    #
    # Every training domain receives equal total weight.
    #
    # This is a baseline domain-generalization strategy.
    # ========================================================

    domain_counts = (
        pd.Series(
            train_domains
        )
        .value_counts()
    )


    weights = np.array([
        1.0 /
        domain_counts[
            d
        ]
        for d in train_domains
    ])


    weights = (
        weights
        /
        weights.mean()
    )


    balanced = make_model()

    balanced.fit(
        X_train,
        y_train,
        sample_weight=weights
    )

    balanced_pred = balanced.predict(
        X_test
    )


    balanced_r2 = r2_score(
        y_test,
        balanced_pred
    )

    balanced_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            balanced_pred
        )
    )

    balanced_mae = mean_absolute_error(
        y_test,
        balanced_pred
    )


    # ========================================================
    # TEST PERFORMANCE BY PREDICTOR-SPACE DOMAIN
    # ========================================================

    for domain in sorted(
        np.unique(
            test_domains
        )
    ):

        mask = (
            test_domains
            == domain
        )

        if mask.sum() < 3:

            continue


        domain_y = (
            y_test.to_numpy()[
                mask
            ]
        )


        erm_domain_pred = (
            erm_pred[
                mask
            ]
        )


        balanced_domain_pred = (
            balanced_pred[
                mask
            ]
        )


        domain_results.append({

            "Fold":
                fold,

            "Domain":
                int(domain),

            "Test_Observations":
                int(mask.sum()),

            "Training_Domain_Count":
                int(
                    domain_counts.get(
                        domain,
                        0
                    )
                ),

            "ERM_R2":
                r2_score(
                    domain_y,
                    erm_domain_pred
                )
                if len(domain_y) >= 2
                else np.nan,

            "ERM_RMSE":
                np.sqrt(
                    mean_squared_error(
                        domain_y,
                        erm_domain_pred
                    )
                ),

            "ERM_MAE":
                mean_absolute_error(
                    domain_y,
                    erm_domain_pred
                ),

            "Balanced_R2":
                r2_score(
                    domain_y,
                    balanced_domain_pred
                )
                if len(domain_y) >= 2
                else np.nan,

            "Balanced_RMSE":
                np.sqrt(
                    mean_squared_error(
                        domain_y,
                        balanced_domain_pred
                    )
                ),

            "Balanced_MAE":
                mean_absolute_error(
                    domain_y,
                    balanced_domain_pred
                )
        })


    # ========================================================
    # PRINT OVERALL FOLD RESULTS
    # ========================================================

    print()
    print("STANDARD ERM XGBOOST")

    print(
        "R²   =",
        round(
            erm_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            erm_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            erm_mae,
            6
        )
    )


    print()
    print("DOMAIN-BALANCED XGBOOST")

    print(
        "R²   =",
        round(
            balanced_r2,
            6
        )
    )

    print(
        "RMSE =",
        round(
            balanced_rmse,
            6
        )
    )

    print(
        "MAE  =",
        round(
            balanced_mae,
            6
        )
    )


    print()
    print("Changes:")

    print(
        "ΔR²   =",
        round(
            balanced_r2
            - erm_r2,
            6
        )
    )

    print(
        "ΔRMSE =",
        round(
            balanced_rmse
            - erm_rmse,
            6
        )
    )

    print(
        "ΔMAE  =",
        round(
            balanced_mae
            - erm_mae,
            6
        )
    )


    # ========================================================
    # STORE OVERALL RESULTS
    # ========================================================

    erm_results.append({

        "Fold":
            fold,

        "R2":
            erm_r2,

        "RMSE":
            erm_rmse,

        "MAE":
            erm_mae
    })


    balanced_results.append({

        "Fold":
            fold,

        "R2":
            balanced_r2,

        "RMSE":
            balanced_rmse,

        "MAE":
            balanced_mae
    })


# ============================================================
# DATAFRAMES
# ============================================================

erm_df = pd.DataFrame(
    erm_results
)

balanced_df = pd.DataFrame(
    balanced_results
)

domain_df = pd.DataFrame(
    domain_results
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "R2",
        "RMSE",
        "MAE"
    ],

    "ERM_Mean": [

        erm_df["R2"].mean(),

        erm_df["RMSE"].mean(),

        erm_df["MAE"].mean()
    ],

    "Balanced_Mean": [

        balanced_df["R2"].mean(),

        balanced_df["RMSE"].mean(),

        balanced_df["MAE"].mean()
    ],

    "ERM_SD": [

        erm_df["R2"].std(
            ddof=1
        ),

        erm_df["RMSE"].std(
            ddof=1
        ),

        erm_df["MAE"].std(
            ddof=1
        )
    ],

    "Balanced_SD": [

        balanced_df["R2"].std(
            ddof=1
        ),

        balanced_df["RMSE"].std(
            ddof=1
        ),

        balanced_df["MAE"].std(
            ddof=1
        )
    ]
})


# ============================================================
# PAIRED FOLD DIFFERENCES
# ============================================================

paired = erm_df.merge(
    balanced_df,
    on="Fold",
    suffixes=(
        "_ERM",
        "_Balanced"
    )
)


paired["Delta_R2"] = (
    paired["R2_Balanced"]
    -
    paired["R2_ERM"]
)

paired["Delta_RMSE"] = (
    paired["RMSE_Balanced"]
    -
    paired["RMSE_ERM"]
)

paired["Delta_MAE"] = (
    paired["MAE_Balanced"]
    -
    paired["MAE_ERM"]
)


# ============================================================
# DOMAIN SUMMARY
# ============================================================

domain_summary = (

    domain_df

    .groupby("Domain")

    .agg(

        Total_Test_Observations=(
            "Test_Observations",
            "sum"
        ),

        Mean_Training_Domain_Count=(
            "Training_Domain_Count",
            "mean"
        ),

        ERM_MAE=(
            "ERM_MAE",
            "mean"
        ),

        Balanced_MAE=(
            "Balanced_MAE",
            "mean"
        ),

        ERM_RMSE=(
            "ERM_RMSE",
            "mean"
        ),

        Balanced_RMSE=(
            "Balanced_RMSE",
            "mean"
        )
    )

    .reset_index()
)


domain_summary["MAE_Change"] = (
    domain_summary["Balanced_MAE"]
    -
    domain_summary["ERM_MAE"]
)


# ============================================================
# PRINT FINAL
# ============================================================

print()
print("============================================================")
print("FINAL DOMAIN-GENERALIZATION SUMMARY")
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


print()
print("============================================================")
print("PER-DOMAIN TEST PERFORMANCE")
print("============================================================")

print(
    domain_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

erm_df.to_csv(
    "YCEO_PredictorDomain_ERM.csv",
    index=False
)

balanced_df.to_csv(
    "YCEO_PredictorDomain_Balanced.csv",
    index=False
)

summary.to_csv(
    "YCEO_PredictorDomain_Summary.csv",
    index=False
)

paired.to_csv(
    "YCEO_PredictorDomain_Paired.csv",
    index=False
)

domain_df.to_csv(
    "YCEO_PredictorDomain_AllDomainResults.csv",
    index=False
)

domain_summary.to_csv(
    "YCEO_PredictorDomain_DomainSummary.csv",
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_PredictorDomain_ERM.csv"
)

print(
    "YCEO_PredictorDomain_Balanced.csv"
)

print(
    "YCEO_PredictorDomain_Summary.csv"
)

print(
    "YCEO_PredictorDomain_Paired.csv"
)

print(
    "YCEO_PredictorDomain_AllDomainResults.csv"
)

print(
    "YCEO_PredictorDomain_DomainSummary.csv"
)

print()
print(
    "Predictor-space domain generalization complete."
)