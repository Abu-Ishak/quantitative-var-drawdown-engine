"""Basel Market Risk VaR Model Validation (Kupiec POF, Christoffersen & Traffic Light).

Implements regulatory backtesting checks for rolling 99% and 95% VaR forecasts:
- Kupiec (1995) Proportion of Failures (POF) Likelihood Ratio Test (chi^2_1)
- Christoffersen (1998) Independence Test for exception clustering
- Basel III 250-day 99% VaR Traffic-Light Zone classification (Green/Yellow/Red)
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _chi2_sf_df1(x: float) -> float:
    """Survival function (p-value) for chi-square(df=1): P(X >= x) = erfc(sqrt(x/2))."""
    if x <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def kupiec_pof_test(exceptions: np.ndarray, alpha: float = 0.01) -> Dict[str, object]:
    """Kupiec (1995) Proportion of Failures Likelihood Ratio Test."""
    n = int(len(exceptions))
    x = int(np.sum(exceptions))
    p_hat = x / max(n, 1)

    if x == 0:
        lr = -2.0 * (n * math.log(max(1.0 - alpha, 1e-12)))
    elif x == n:
        lr = -2.0 * (n * math.log(max(alpha, 1e-12)))
    else:
        log_null = x * math.log(alpha) + (n - x) * math.log(1.0 - alpha)
        log_alt = x * math.log(p_hat) + (n - x) * math.log(1.0 - p_hat)
        lr = -2.0 * (log_null - log_alt)

    p_val = _chi2_sf_df1(lr)
    return {
        "n_obs": n,
        "exceptions": x,
        "expected": round(n * alpha, 2),
        "rate": p_hat,
        "LR_POF": lr,
        "p_value": p_val,
        "pass_95pct": p_val >= 0.05,
    }


def christoffersen_independence_test(exceptions: np.ndarray) -> Dict[str, float]:
    """Test whether VaR breaches occur independently or cluster together."""
    exc = exceptions.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(exc)):
        prev, curr = exc[i - 1], exc[i]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)

    def _safe_log(p: float) -> float:
        return math.log(min(max(p, 1e-12), 1.0 - 1e-12))

    ll_indep = (n00 + n10) * _safe_log(1.0 - pi) + (n01 + n11) * _safe_log(pi)
    ll_markov = (
        n00 * _safe_log(1.0 - pi0)
        + n01 * _safe_log(pi0)
        + n10 * _safe_log(1.0 - pi1)
        + n11 * _safe_log(pi1)
    )
    lr_ind = max(0.0, -2.0 * (ll_indep - ll_markov))
    return {"LR_IND": lr_ind, "p_value": _chi2_sf_df1(lr_ind)}


def basel_traffic_light(exceptions_250d: int) -> str:
    """Basel III 250-day 99% VaR backtesting zone classification."""
    if exceptions_250d <= 4:
        return "GREEN (0-4 breaches)"
    if exceptions_250d <= 9:
        return "YELLOW (5-9 breaches)"
    return "RED (10+ breaches)"
