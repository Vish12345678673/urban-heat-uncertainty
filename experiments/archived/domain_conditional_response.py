import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from scipy.stats import pearsonr, spearmanr


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42

N_DOMAINS = 5

MIN_DOMAIN_SIZE = 100

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


X = df[FEATURES].copy()

y = df[TARGET].copy()


print()
print("============================================================")
print("DOMAIN-CONDITIONAL RESPONSE ANALYSIS")
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
    "Domains:",
    N_DOMAINS
)


# ============================================================
# STANDARDIZE PREDICTORS
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# PCA
# ============================================================

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


explained = pca.explained_variance_ratio_

print()
print("PCA explained variance:")

for i, value in enumerate(
    explained,
    start=1
):

    print(
        f"PC{i}:",
        round(
            value,
            4
        )
    )


# ============================================================
# K-MEANS ENVIRONMENTAL-URBAN DOMAINS
# ============================================================

kmeans = KMeans(
    n_clusters=N_DOMAINS,
    n_init=20,
    random_state=RANDOM_STATE
)

domains = kmeans.fit_predict(
    X_pca
)

df["Domain"] = domains


print()
print("============================================================")
print("DOMAIN SIZES")
print("============================================================")

print(
    df["Domain"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# DOMAIN DESCRIPTIVE STATISTICS
# ============================================================

domain_profile = (
    df
    .groupby("Domain")[FEATURES + [TARGET]]
    .mean()
    .reset_index()
)


print()
print("============================================================")
print("DOMAIN MEANS")
print("============================================================")

print(
    domain_profile.to_string(
        index=False
    )
)


# ============================================================
# DOMAIN TARGET DISTRIBUTION
# ============================================================

target_profile = (
    df
    .groupby("Domain")[TARGET]
    .agg(
        Observations="count",
        Mean="mean",
        SD="std",
        Median="median",
        Min="min",
        Max="max"
    )
    .reset_index()
)


print()
print("============================================================")
print("DOMAIN TARGET DISTRIBUTIONS")
print("============================================================")

print(
    target_profile.to_string(
        index=False
    )
)


# ============================================================
# WITHIN-DOMAIN CORRELATIONS
# ============================================================

correlation_rows = []


print()
print("============================================================")
print("WITHIN-DOMAIN SPEARMAN CORRELATIONS")
print("============================================================")


for domain in sorted(
    df["Domain"].unique()
):

    temp = df[
        df["Domain"] == domain
    ].copy()


    print()
    print(
        "DOMAIN",
        domain
    )


    if len(temp) < MIN_DOMAIN_SIZE:

        print(
            "Too few observations."
        )

        continue


    row = {
        "Domain":
            domain,

        "Observations":
            len(temp)
    }


    for feature in FEATURES:

        r, p = spearmanr(
            temp[feature],
            temp[TARGET]
        )


        print(
            feature,
            ": r =",
            round(
                r,
                4
            ),
            "p =",
            f"{p:.3e}"
        )


        row[
            f"{feature}_Spearman_r"
        ] = r

        row[
            f"{feature}_Spearman_p"
        ] = p


    correlation_rows.append(
        row
    )


correlation_df = pd.DataFrame(
    correlation_rows
)


# ============================================================
# DOMAIN-SPECIFIC MODELS
# ============================================================

model_rows = []


print()
print("============================================================")
print("DOMAIN-SPECIFIC MODEL PERFORMANCE")
print("============================================================")


for domain in sorted(
    df["Domain"].unique()
):

    temp = df[
        df["Domain"] == domain
    ].copy()


    if len(temp) < MIN_DOMAIN_SIZE:

        continue


    X_domain = temp[
        FEATURES
    ]

    y_domain = temp[
        TARGET
    ]


    # --------------------------------------------------------
    # Internal random split ONLY within this domain.
    # This is descriptive, not the main spatial validation.
    # --------------------------------------------------------

    split = int(
        len(temp) * 0.8
    )


    rng = np.random.default_rng(
        RANDOM_STATE
        + domain
    )


    indices = np.arange(
        len(temp)
    )

    rng.shuffle(
        indices
    )


    train_local = indices[
        :split
    ]

    test_local = indices[
        split:
    ]


    X_tr = X_domain.iloc[
        train_local
    ]

    X_te = X_domain.iloc[
        test_local
    ]

    y_tr = y_domain.iloc[
        train_local
    ]

    y_te = y_domain.iloc[
        test_local
    ]


    model = GradientBoostingRegressor(

        n_estimators=300,

        learning_rate=0.03,

        max_depth=3,

        min_samples_leaf=15,

        random_state=RANDOM_STATE
    )


    model.fit(
        X_tr,
        y_tr
    )


    pred = model.predict(
        X_te
    )


    r2 = r2_score(
        y_te,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_te,
            pred
        )
    )

    mae = mean_absolute_error(
        y_te,
        pred
    )


    model_rows.append({

        "Domain":
            domain,

        "Observations":
            len(temp),

        "R2":
            r2,

        "RMSE":
            rmse,

        "MAE":
            mae
    })


    print()
    print(
        "Domain",
        domain
    )

    print(
        "Observations:",
        len(temp)
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


domain_model_df = pd.DataFrame(
    model_rows
)


# ============================================================
# GLOBAL MODEL VS DOMAIN-SPECIFIC MODELS
# ============================================================

global_model = GradientBoostingRegressor(

    n_estimators=300,

    learning_rate=0.03,

    max_depth=3,

    min_samples_leaf=15,

    random_state=RANDOM_STATE
)


# Random 80/20 split for descriptive comparison only.
rng = np.random.default_rng(
    RANDOM_STATE
)

indices = np.arange(
    len(df)
)

rng.shuffle(
    indices
)

split = int(
    0.8 * len(df)
)

global_train = indices[
    :split
]

global_test = indices[
    split:
]


global_model.fit(
    X.iloc[global_train],
    y.iloc[global_train]
)

global_pred = global_model.predict(
    X.iloc[global_test]
)


global_r2 = r2_score(
    y.iloc[global_test],
    global_pred
)

global_rmse = np.sqrt(
    mean_squared_error(
        y.iloc[global_test],
        global_pred
    )
)

global_mae = mean_absolute_error(
    y.iloc[global_test],
    global_pred
)


print()
print("============================================================")
print("GLOBAL DESCRIPTIVE MODEL")
print("============================================================")

print(
    "R²   =",
    round(
        global_r2,
        6
    )
)

print(
    "RMSE =",
    round(
        global_rmse,
        6
    )
)

print(
    "MAE  =",
    round(
        global_mae,
        6
    )
)


# ============================================================
# PREDICTOR EFFECT HETEROGENEITY
# ============================================================
#
# Fit simple standardized linear response models separately
# within each domain.
#
# Coefficient differences provide a direct descriptive measure
# of whether the predictor-target mapping changes between
# environmental domains.
# ============================================================

from sklearn.linear_model import LinearRegression


coef_rows = []


print()
print("============================================================")
print("DOMAIN-SPECIFIC STANDARDIZED LINEAR COEFFICIENTS")
print("============================================================")


for domain in sorted(
    df["Domain"].unique()
):

    temp = df[
        df["Domain"] == domain
    ].copy()


    if len(temp) < MIN_DOMAIN_SIZE:

        continue


    scaler_domain = StandardScaler()

    X_domain_scaled = (
        scaler_domain.fit_transform(
            temp[FEATURES]
        )
    )


    y_domain = (
        temp[TARGET]
    )


    linear = LinearRegression()

    linear.fit(
        X_domain_scaled,
        y_domain
    )


    print()
    print(
        "Domain",
        domain
    )


    row = {
        "Domain":
            domain,

        "Observations":
            len(temp)
    }


    for feature, coefficient in zip(
        FEATURES,
        linear.coef_
    ):

        print(
            feature,
            ":",
            round(
                coefficient,
                6
            )
        )


        row[
            feature
        ] = coefficient


    coef_rows.append(
        row
    )


coef_df = pd.DataFrame(
    coef_rows
)


# ============================================================
# COEFFICIENT VARIABILITY
# ============================================================

coefficient_variability = []


for feature in FEATURES:

    if feature not in coef_df.columns:

        continue


    values = coef_df[
        feature
    ].to_numpy()


    coefficient_variability.append({

        "Feature":
            feature,

        "Mean_Domain_Coefficient":
            np.mean(values),

        "SD_Domain_Coefficient":
            np.std(
                values,
                ddof=1
            ),

        "Min_Domain_Coefficient":
            np.min(values),

        "Max_Domain_Coefficient":
            np.max(values),

        "Range":
            np.max(values)
            -
            np.min(values)
    })


variability_df = pd.DataFrame(
    coefficient_variability
)


print()
print("============================================================")
print("COEFFICIENT VARIABILITY")
print("============================================================")

print(
    variability_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

domain_profile.to_csv(
    "YCEO_DomainConditional_DomainProfiles.csv",
    index=False
)

target_profile.to_csv(
    "YCEO_DomainConditional_TargetProfiles.csv",
    index=False
)

correlation_df.to_csv(
    "YCEO_DomainConditional_WithinDomainCorrelations.csv",
    index=False
)

domain_model_df.to_csv(
    "YCEO_DomainConditional_DomainModels.csv",
    index=False
)

coef_df.to_csv(
    "YCEO_DomainConditional_DomainCoefficients.csv",
    index=False
)

variability_df.to_csv(
    "YCEO_DomainConditional_CoefficientVariability.csv",
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
    "YCEO_DomainConditional_DomainProfiles.csv"
)

print(
    "YCEO_DomainConditional_TargetProfiles.csv"
)

print(
    "YCEO_DomainConditional_WithinDomainCorrelations.csv"
)

print(
    "YCEO_DomainConditional_DomainModels.csv"
)

print(
    "YCEO_DomainConditional_DomainCoefficients.csv"
)

print(
    "YCEO_DomainConditional_CoefficientVariability.csv"
)

print()
print(
    "Domain-conditional response analysis complete."
)