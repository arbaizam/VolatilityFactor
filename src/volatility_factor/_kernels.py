"""Numba-compiled simulation kernels."""

from __future__ import annotations

import numpy as np
from numba import jit


@jit(nopython=True)
def nday_price_sim(
    horizon: int,
    shocks: np.ndarray,
    coefs: np.ndarray,
    mean: np.ndarray,
    resid_start: float,
    sigma_start: float,
    nsims: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate GARCH(1,1) paths.

    Args:
        horizon: number of periods to simulate.
        shocks: ``(nsims, horizon)`` array of standardized shocks.
        coefs: ``[omega, alpha, beta]``.
        mean: zero-dimensional array holding the constant mean.
        resid_start: starting residual.
        sigma_start: starting conditional volatility.
        nsims: number of paths.

    Returns:
        ``(y, residuals, sigma)``, each ``(nsims, horizon)``.
    """
    resid_sim = np.empty(shape=(nsims, horizon + 1), dtype=np.float64)
    sigma_sim = np.empty(shape=(nsims, horizon + 1), dtype=np.float64)
    y_sim = np.empty(shape=(nsims, horizon), dtype=np.float64)

    for i in range(np.int64(nsims)):
        for j in range(horizon):
            resid_sim[i, 0] = resid_start
            sigma_sim[i, 0] = sigma_start
            yyy = np.array([1, resid_sim[i, j] ** 2, sigma_sim[i, j] ** 2])
            sigma_sim[i, j + 1] = np.sqrt(np.dot(coefs, yyy))
            resid_sim[i, j + 1] = sigma_sim[i, j + 1] * shocks[i, j]
            y_sim[i, j] = mean + resid_sim[i, j + 1]
    return y_sim, resid_sim[:, 1:], sigma_sim[:, 1:]
