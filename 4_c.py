import numpy as np
from scipy.stats import norm
import pandas as pd


def normal_ci(x, alpha=0.05):
    n = len(x)
    lam_hat = np.mean(x)
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(lam_hat / n)

    lower = lam_hat - z * se
    upper = lam_hat + z * se


    lower = max(0, lower)
    return lower, upper



def bootstrap_ci(x, B=1000, alpha=0.05):
    n = len(x)
    lam_hat = np.mean(x)

    bootestimates = np.empty(B)

    for b in range(B):
        x_star = np.random.poisson(lam_hat, size=n)
        bootestimates[b] = np.mean(x_star)

    lower = np.quantile(bootestimates, alpha / 2)
    upper = np.quantile(bootestimates, 1 - alpha / 2)
    return lower, upper



def coverage(n, lam_true=1, R=1000, B=1000, alpha=0.05):
    normal_cover = 0
    bootstrap_cover = 0

    for r in range(R):

        x = np.random.poisson(lam_true, size=n)


        l1, u1 = normal_ci(x, alpha=alpha)
        if l1 <= lam_true <= u1:
            normal_cover += 1


        l2, u2 = bootstrap_ci(x, B=B, alpha=alpha)
        if l2 <= lam_true <= u2:
            bootstrap_cover += 1

    return {
        "n": n,
        "Normal CI Coverage": normal_cover / R,
        "Bootstrap CI Coverage": bootstrap_cover / R
    }


np.random.seed(12345)

sample_sizes = [5, 10, 100, 1000]
results = []

for n in sample_sizes:
    results.append(coverage(n, lam_true=1, R=1000, B=1000, alpha=0.05))

df = pd.DataFrame(results)
print(df)