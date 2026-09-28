import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


# =====================================================
# LOAD FINAL PREDICTIONS
# =====================================================

df = pd.read_csv(
    "YCEO_UHI_Final_Predictions.csv"
)

coords = df[
    ["Lat", "Lon"]
].to_numpy()

residuals = df[
    "Residual"
].to_numpy()

n = len(residuals)

z = residuals - residuals.mean()


# =====================================================
# DISTANCE MATRIX
# =====================================================

print()
print("======================================")
print("MORAN'S I SENSITIVITY ANALYSIS")
print("======================================")

print(
    "Observations:",
    n
)

print(
    "Calculating spatial distances..."
)

distances = cdist(
    coords,
    coords,
    metric="euclidean"
)

np.fill_diagonal(
    distances,
    np.inf
)


# =====================================================
# FUNCTION
# =====================================================

def moran_test(k, permutations=999):

    print()
    print(
        "Testing k =",
        k
    )

    # -----------------------------------------------
    # Nearest neighbors
    # -----------------------------------------------

    neighbors = np.argpartition(
        distances,
        kth=k-1,
        axis=1
    )[:, :k]


    # -----------------------------------------------
    # Row-standardized weights
    # -----------------------------------------------

    W = np.zeros(
        (n, n),
        dtype=np.float64
    )

    for i in range(n):

        W[
            i,
            neighbors[i]
        ] = 1.0 / k


    W_sum = W.sum()


    # -----------------------------------------------
    # Observed Moran's I
    # -----------------------------------------------

    numerator = (
        z @ W @ z
    )

    denominator = (
        z @ z
    )

    observed = (
        (n / W_sum)
        *
        numerator
        /
        denominator
    )


    # -----------------------------------------------
    # Expected Moran's I
    # -----------------------------------------------

    expected = (
        -1 / (n - 1)
    )


    # -----------------------------------------------
    # Permutations
    # -----------------------------------------------

    rng = np.random.default_rng(
        42
    )

    permutation_values = np.empty(
        permutations
    )

    for p in range(permutations):

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


    # -----------------------------------------------
    # P-value
    # -----------------------------------------------

    extreme = np.sum(
        np.abs(
            permutation_values
        )
        >= abs(observed)
    )

    p_value = (
        extreme + 1
    ) / (
        permutations + 1
    )


    # -----------------------------------------------
    # Z-score
    # -----------------------------------------------

    perm_mean = (
        permutation_values.mean()
    )

    perm_std = (
        permutation_values.std(
            ddof=1
        )
    )

    z_score = (
        observed - perm_mean
    ) / perm_std


    print(
        "Moran's I:",
        round(
            observed,
            6
        )
    )

    print(
        "Expected I:",
        round(
            expected,
            6
        )
    )

    print(
        "Permutation Z:",
        round(
            z_score,
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


    return {
        "k": k,
        "Moran_I": observed,
        "Expected_I": expected,
        "Permutation_Mean": perm_mean,
        "Permutation_SD": perm_std,
        "Permutation_Z": z_score,
        "Permutation_p": p_value
    }


# =====================================================
# RUN SENSITIVITY TEST
# =====================================================

results = []

for k in [4, 8, 16]:

    results.append(
        moran_test(
            k=k,
            permutations=999
        )
    )


# =====================================================
# SAVE
# =====================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "YCEO_UHI_Moran_Sensitivity.csv",
    index=False
)


# =====================================================
# FINAL SUMMARY
# =====================================================

print()
print("======================================")
print("SENSITIVITY SUMMARY")
print("======================================")

print(
    results_df.to_string(
        index=False
    )
)

print()
print(
    "Saved:"
)

print(
    "YCEO_UHI_Moran_Sensitivity.csv"
)

print()
print(
    "Moran's I sensitivity analysis complete."
)