import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/YCEO_UHI_Enhanced_ML_Clean_Daytime.csv"

RANDOM_STATE = 42
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

# Main predictors for explicit interaction testing.
TEST_FEATURES = [
    "NDVI",
    "Mean_Temp_C",
    "Mean_Monthly_Precip_mm",
    "Builtup",
    "Population"
]


# ============================================================
# LOAD AND CLEAN
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


print()
print("============================================================")
print("FORMAL DOMAIN × PREDICTOR INTERACTION TEST")
print("============================================================")

print(
    "Observations:",
    len(df)
)

print(
    "Domains:",
    N_DOMAINS
)

print(
    "Predictors tested:",
    ", ".join(TEST_FEATURES)
)


# ============================================================
# CREATE PREDICTOR-SPACE DOMAINS
# ============================================================
#
# Important:
# This is a descriptive formal test on the complete dataset.
# It is NOT the outer spatial-validation benchmark.
#
# Domain labels are generated only from predictors, never from
# Annual_day.
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[FEATURES]
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

df["Domain"] = kmeans.fit_predict(
    X_pca
)


# ============================================================
# STANDARDIZE TEST PREDICTORS
# ============================================================
#
# Standardization makes interaction coefficients comparable.
# ============================================================

for feature in TEST_FEATURES:

    mean = df[feature].mean()

    std = df[feature].std()

    if std < 1e-12:
        std = 1.0

    df[
        f"{feature}_z"
    ] = (
        df[feature]
        - mean
    ) / std


# ============================================================
# DOMAIN SUMMARY
# ============================================================

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
# TEST EACH PREDICTOR
# ============================================================
#
# Model:
#
# Annual_day ~ predictor + Domain + predictor:Domain
#
# Domain is treated as categorical.
#
# The predictor × Domain terms test whether the slope differs
# between domains.
# ============================================================

results = []

full_model_summaries = []


for feature in TEST_FEATURES:

    z_feature = f"{feature}_z"

    print()
    print("============================================================")
    print(
        "TESTING:",
        feature
    )
    print("============================================================")


    formula = (
        f"{TARGET} ~ "
        f"{z_feature} * C(Domain)"
    )


    model = ols(
        formula,
        data=df
    ).fit(
        cov_type="HC3"
    )


    # --------------------------------------------------------
    # Extract interaction terms.
    # Domain 0 is the reference domain.
    # --------------------------------------------------------

    interaction_terms = [
        term
        for term in model.params.index
        if f"{z_feature}:C(Domain)" in term
    ]


    if len(interaction_terms) == 0:

        print(
            "No interaction terms found."
        )

        continue


    # --------------------------------------------------------
    # Individual interaction results.
    # --------------------------------------------------------

    for term in interaction_terms:

        coefficient = model.params[
            term
        ]

        p_value = model.pvalues[
            term
        ]

        ci_low, ci_high = (
            model.conf_int()
            .loc[term]
        )


        domain_number = (
            term
            .split("T.")[-1]
            .replace("]", "")
        )


        print()
        print(
            term
        )

        print(
            "Coefficient:",
            round(
                coefficient,
                6
            )
        )

        print(
            "95% CI:",
            (
                round(
                    ci_low,
                    6
                ),
                round(
                    ci_high,
                    6
                )
            )
        )

        print(
            "p-value:",
            f"{p_value:.6e}"
        )


        results.append({

            "Predictor":
                feature,

            "Interaction_Term":
                term,

            "Comparison_Domain":
                domain_number,

            "Coefficient":
                coefficient,

            "CI_Low":
                ci_low,

            "CI_High":
                ci_high,

            "Raw_p":
                p_value
        })


    # --------------------------------------------------------
    # JOINT TEST:
    #
    # All predictor × domain interaction terms = 0
    #
    # This is the main test for domain-dependent slope.
    # --------------------------------------------------------

    interaction_constraints = ", ".join(
        [
            f"{term} = 0"
            for term in interaction_terms
        ]
    )


    joint_test = model.f_test(
        interaction_constraints
    )


    joint_p = float(
        joint_test.pvalue
    )


    joint_stat = float(
        np.asarray(
            joint_test.fvalue
        ).reshape(-1)[0]
    )


    print()
    print(
        "JOINT INTERACTION TEST"
    )

    print(
        "F-statistic:",
        round(
            joint_stat,
            6
        )
    )

    print(
        "p-value:",
        f"{joint_p:.6e}"
    )


    full_model_summaries.append({

        "Predictor":
            feature,

        "Joint_F":
            joint_stat,

        "Joint_p":
            joint_p,

        "R2":
            model.rsquared,

        "Adjusted_R2":
            model.rsquared_adj,

        "N_Interaction_Terms":
            len(interaction_terms)
    })


# ============================================================
# MULTIPLE TESTING CORRECTION
# ============================================================
#
# Five predictor-level joint tests are performed.
# Benjamini-Hochberg FDR correction is applied.
# ============================================================

joint_df = pd.DataFrame(
    full_model_summaries
)


if len(joint_df) > 0:

    reject, corrected_p, _, _ = (
        multipletests(
            joint_df["Joint_p"],
            alpha=0.05,
            method="fdr_bh"
        )
    )

    joint_df[
        "FDR_Adjusted_p"
    ] = corrected_p

    joint_df[
        "Significant_FDR_0.05"
    ] = reject


# ============================================================
# CORRECT INDIVIDUAL INTERACTION TESTS
# ============================================================

interaction_df = pd.DataFrame(
    results
)


if len(interaction_df) > 0:

    reject_i, corrected_i, _, _ = (
        multipletests(
            interaction_df["Raw_p"],
            alpha=0.05,
            method="fdr_bh"
        )
    )

    interaction_df[
        "FDR_Adjusted_p"
    ] = corrected_i

    interaction_df[
        "Significant_FDR_0.05"
    ] = reject_i


# ============================================================
# DOMAIN-SPECIFIC SLOPES
# ============================================================
#
# For interpretability, explicitly calculate the slope for each
# predictor in each domain.
# ============================================================

slope_rows = []


print()
print("============================================================")
print("DOMAIN-SPECIFIC SLOPES")
print("============================================================")


for feature in TEST_FEATURES:

    z_feature = f"{feature}_z"


    for domain in sorted(
        df["Domain"].unique()
    ):

        temp = df[
            df["Domain"] == domain
        ]


        if len(temp) < 30:
            continue


        slope_model = ols(
            f"{TARGET} ~ {z_feature}",
            data=temp
        ).fit(
            cov_type="HC3"
        )


        slope = slope_model.params[
            z_feature
        ]

        p_value = slope_model.pvalues[
            z_feature
        ]

        ci_low, ci_high = (
            slope_model.conf_int()
            .loc[z_feature]
        )


        slope_rows.append({

            "Predictor":
                feature,

            "Domain":
                domain,

            "Observations":
                len(temp),

            "Standardized_Slope":
                slope,

            "CI_Low":
                ci_low,

            "CI_High":
                ci_high,

            "p_value":
                p_value
        })


        print(
            f"{feature:30s}"
            f" Domain {domain}: "
            f"slope={slope:.6f} "
            f"CI=[{ci_low:.6f}, "
            f"{ci_high:.6f}] "
            f"p={p_value:.3e}"
        )


slope_df = pd.DataFrame(
    slope_rows
)


# ============================================================
# RANGE OF DOMAIN SLOPES
# ============================================================

variability_rows = []


for feature in TEST_FEATURES:

    temp = slope_df[
        slope_df["Predictor"] == feature
    ]


    if len(temp) == 0:
        continue


    values = temp[
        "Standardized_Slope"
    ].to_numpy()


    variability_rows.append({

        "Predictor":
            feature,

        "Mean_Slope":
            values.mean(),

        "SD_Slope":
            values.std(
                ddof=1
            ),

        "Min_Slope":
            values.min(),

        "Max_Slope":
            values.max(),

        "Slope_Range":
            values.max()
            - values.min()
    })


variability_df = pd.DataFrame(
    variability_rows
)


print()
print("============================================================")
print("DOMAIN SLOPE VARIABILITY")
print("============================================================")

print(
    variability_df.to_string(
        index=False
    )
)


# ============================================================
# FINAL JOINT RESULTS
# ============================================================

print()
print("============================================================")
print("JOINT DOMAIN × PREDICTOR TESTS")
print("============================================================")

print(
    joint_df.to_string(
        index=False
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

if len(joint_df) > 0:

    significant_predictors = (
        joint_df[
            joint_df[
                "Significant_FDR_0.05"
            ]
        ]["Predictor"]
        .tolist()
    )

else:

    significant_predictors = []


print()
print("============================================================")
print("INTERPRETATION")
print("============================================================")


if significant_predictors:

    print(
        "Evidence of significant domain-dependent "
        "predictor effects was found after FDR correction:"
    )

    for predictor in significant_predictors:

        print(
            " -",
            predictor
        )

else:

    print(
        "No predictor showed a statistically significant "
        "domain interaction after FDR correction."
    )


# ============================================================
# SAVE
# ============================================================

joint_df.to_csv(
    "YCEO_DomainInteraction_JointTests.csv",
    index=False
)

interaction_df.to_csv(
    "YCEO_DomainInteraction_IndividualTests.csv",
    index=False
)

slope_df.to_csv(
    "YCEO_DomainInteraction_DomainSlopes.csv",
    index=False
)

variability_df.to_csv(
    "YCEO_DomainInteraction_SlopeVariability.csv",
    index=False
)


# ============================================================
# FULL REGRESSION SUMMARIES
# ============================================================

for feature in TEST_FEATURES:

    z_feature = f"{feature}_z"

    model = ols(
        f"{TARGET} ~ {z_feature} * C(Domain)",
        data=df
    ).fit(
        cov_type="HC3"
    )


    with open(
        f"YCEO_DomainInteraction_{feature}_Model.txt",
        "w"
    ) as f:

        f.write(
            model.summary().as_text()
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("============================================================")
print("FILES CREATED")
print("============================================================")

print(
    "YCEO_DomainInteraction_JointTests.csv"
)

print(
    "YCEO_DomainInteraction_IndividualTests.csv"
)

print(
    "YCEO_DomainInteraction_DomainSlopes.csv"
)

print(
    "YCEO_DomainInteraction_SlopeVariability.csv"
)

print()
print(
    "Domain interaction analysis complete."
)