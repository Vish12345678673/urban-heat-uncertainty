import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm

# =====================================================
# LOAD FINAL PREDICTIONS
# =====================================================

df = pd.read_csv(
    "YCEO_UHI_Final_Predictions.csv"
)

print()
print("======================================")
print("MORAN'S I SPATIAL AUTOCORRELATION TEST")
print("======================================")

print(
    "Observations:",
    len(df)
)


# =====================================================
# COORDINATES AND RESIDUALS
# =====================================================

coords = df[
    ["Lat", "Lon"]
].to_numpy()

residuals = df[
    "Residual"
].to_numpy()

n = len(residuals)

z = residuals - residuals.mean()


# =====================================================
# K-NEAREST-NEIGHBOR SPATIAL WEIGHTS
# =====================================================
#
# We use k = 8 nearest spatial neighbors.
#
# Row-standardization means every observation has
# total spatial weight = 1.
# =====================================================

k = 8

print()
print(
    "Spatial weights: ",
    k,
    "nearest neighbors"
)

# Euclidean distance in latitude/longitude degrees.
# This is adequate for this diagnostic because the
# objective is residual-neighborhood association,
# not physical distance estimation.
distances = cdist(
    coords,
    coords,
    metric="euclidean"
)

np.fill_diagonal(
    distances,
    np.inf
)

neighbors = np.argpartition(
    distances,
    kth=k-1,
    axis=1
)[:, :k]


# =====================================================
# BUILD ROW-STANDARDIZED WEIGHTS
# =====================================================

W = np.zeros(
    (n, n),
    dtype=np.float64
)

for i in range(n):

    W[
        i,
        neighbors[i]
    ] = 1.0 / k


# =====================================================
# MORAN'S I
# =====================================================

W_sum = W.sum()

numerator = (
    z @ W @ z
)

denominator = (
    z @ z
)

moran_i = (
    (n / W_sum)
    *
    (numerator / denominator)
)


# =====================================================
# EXPECTED MORAN'S I
# =====================================================

expected_i = (
    -1 / (n - 1)
)


# =====================================================
# APPROXIMATE RANDOMIZATION Z-SCORE
# =====================================================

# Permutation test
# -----------------------------------------------------
# 999 random permutations of residuals.
# This avoids relying only on the theoretical
# normal approximation.
# -----------------------------------------------------

print()
print(
    "Running 999 permutations..."
)

rng = np.random.default_rng(
    42
)

permutation_values = np.empty(
    999
)

for p in range(999):

    permuted = rng.permutation(
        z
    )

    permutation_values[p] = (
        (n / W_sum)
        *
        (
            permuted
            @ W
            @ permuted
        )
        /
        (
            permuted
            @ permuted
        )
    )


# =====================================================
# PERMUTATION P-VALUE
# =====================================================

extreme_count = (
    np.sum(
        np.abs(
            permutation_values
        )
        >= abs(moran_i)
    )
)

p_value = (
    extreme_count + 1
) / (
    len(permutation_values) + 1
)


# =====================================================
# PERMUTATION Z-SCORE
# =====================================================

perm_mean = (
    permutation_values.mean()
)

perm_std = (
    permutation_values.std(
        ddof=1
    )
)

permutation_z = (
    (moran_i - perm_mean)
    / perm_std
)


# =====================================================
# INTERPRETATION
# =====================================================

if p_value < 0.05:

    if moran_i > expected_i:

        interpretation = (
            "Significant positive spatial "
            "autocorrelation: residuals are "
            "spatially clustered."
        )

    else:

        interpretation = (
            "Significant negative spatial "
            "autocorrelation: neighboring "
            "residuals tend to differ."
        )

else:

    interpretation = (
        "No statistically significant spatial "
        "autocorrelation detected."
    )


# =====================================================
# PRINT RESULTS
# =====================================================

print()
print("======================================")
print("RESULT")
print("======================================")

print(
    "Moran's I:",
    round(
        moran_i,
        6
    )
)

print(
    "Expected I:",
    round(
        expected_i,
        6
    )
)

print(
    "Permutation mean:",
    round(
        perm_mean,
        6
    )
)

print(
    "Permutation SD:",
    round(
        perm_std,
        6
    )
)

print(
    "Permutation Z:",
    round(
        permutation_z,
        4
    )
)

print(
    "Permutation p-value:",
    round(
        p_value,
        6
    )
)

print()
print(
    "Interpretation:"
)

print(
    interpretation
)


# =====================================================
# SAVE RESULTS
# =====================================================

result = pd.DataFrame([
    {
        "Moran_I": moran_i,
        "Expected_I": expected_i,
        "Permutation_Mean": perm_mean,
        "Permutation_SD": perm_std,
        "Permutation_Z": permutation_z,
        "Permutation_p": p_value,
        "Neighbors_k": k,
        "Observations": n,
        "Interpretation": interpretation
    }
])

result.to_csv(
    "YCEO_UHI_Morans_I_Result.csv",
    index=False
)

print()
print(
    "Saved:"
)

print(
    "YCEO_UHI_Morans_I_Result.csv"
)

print()
print(
    "Moran's I analysis complete."
)