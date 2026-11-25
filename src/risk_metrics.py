"""Parametric, Historical, Cornish-Fisher & Monte Carlo VaR/CVaR Risk Metrics.

Operationalizes empirical tail-risk and drawdown methodologies from Abu Ishak's
B.Sc. Statistics Honours Thesis ('Zeroing In on Risk: A Quantitative Analysis of
Challenges and Solutions in Algorithmic Market Trading').
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation for the standard normal inverse CDF."""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def sample_moments(returns: np.ndarray) -> Dict[str, float]:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    centered = (returns - mu) / max(sigma, 1e-12)
    skew = float(np.mean(centered ** 3))
    excess_kurt = float(np.mean(centered ** 4) - 3.0)
    return {"mean": mu, "std": sigma, "skew": skew, "excess_kurtosis": excess_kurt}


def parametric_var_cvar(returns: np.ndarray, alpha: float = 0.01) -> Dict[str, float]:
    """Gaussian Variance-Covariance VaR and Expected Shortfall (reported as positive loss %)."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z = norm_ppf(alpha)
    var = -(mu + z * sigma)
    cvar = -mu + sigma * (norm_pdf(z) / alpha)
    return {"VaR": var, "CVaR": cvar}


def historical_var_cvar(returns: np.ndarray, alpha: float = 0.01) -> Dict[str, float]:
    """Non-parametric empirical quantile VaR and tail mean CVaR."""
    q = float(np.quantile(returns, alpha))
    tail = returns[returns <= q]
    return {"VaR": -q, "CVaR": float(-np.mean(tail)) if len(tail) > 0 else -q}


def cornish_fisher_var(returns: np.ndarray, alpha: float = 0.01) -> float:
    """Skewness- and kurtosis-adjusted Cornish-Fisher Value-at-Risk."""
    m = sample_moments(returns)
    z = norm_ppf(alpha)
    s = m["skew"]
    k = m["excess_kurtosis"]
    z_cf = (
        z
        + ((z ** 2 - 1.0) * s) / 6.0
        + ((z ** 3 - 3.0 * z) * k) / 24.0
        - ((2.0 * (z ** 3) - 5.0 * z) * (s ** 2)) / 36.0
    )
    return -(m["mean"] + z_cf * m["std"])


def monte_carlo_student_t_var_cvar(
    returns: np.ndarray,
    alpha: float = 0.01,
    df: float = 4.5,
    n_sims: int = 50_000,
    seed: int = 7,
) -> Dict[str, float]:
    """Monte Carlo fat-tail Student-t VaR & CVaR calibrated to empirical scale."""
    rng = np.random.default_rng(seed)
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    scale = sigma * math.sqrt((df - 2.0) / df)
    sim = mu + scale * rng.standard_t(df, size=n_sims)
    q = float(np.quantile(sim, alpha))
    tail = sim[sim <= q]
    return {"VaR": -q, "CVaR": float(-np.mean(tail))}


def drawdown_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Compute Max Drawdown, Ulcer Index, Max Underwater Duration, and Tail Ratio."""
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))
    ulcer = float(np.sqrt(np.mean((dd * 100.0) ** 2)))
    max_dur = 0
    cur_dur = 0
    for val in dd:
        if val < 0.0:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    p95 = float(np.quantile(returns, 0.95))
    p05 = abs(float(np.quantile(returns, 0.05)))
    tail_ratio = p95 / max(p05, 1e-8)
    return {
        "MaxDD": max_dd,
        "UlcerIndex": ulcer,
        "MaxUnderwaterDays": float(max_dur),
        "TailRatio_95_05": tail_ratio,
    }
