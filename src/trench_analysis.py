"""Analysis helpers for the Combine -> in-game -> outcome chain (trench).

All functions are pure pandas/numpy so they run locally and inside the final
Kaggle notebook unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OL_POS = ["T", "G", "C"]
DL_POS = ["DT", "DE", "OLB", "NT"]

GAME_METRICS = ["depth_05", "depth_10", "depth_15", "lat_abs_10", "s_05", "s_10", "smax_10", "smax_15",
                "amax_05", "t_s2", "t_d1", "t_d2", "t_d1_d2", "s_at_d1", "set_angle_10", "orel_00",
                "orel_05", "orel_10"]


# ----------------------------------------------------------------------------
# loading / cleaning
# ----------------------------------------------------------------------------
def load_game_features(path) -> pd.DataFrame:
    g = pd.read_csv(path, low_memory=False)
    g = g[(g.play_nullified_by_penalty != "Y") & (g.quarter <= 4)]
    # sanity: player should be still before the snap; drop mis-tagged snaps
    g = g[(g.pre_s.isna()) | (g.pre_s < 1.5)]
    g = g[g.depth_10.notna()]
    g["pressure_allowed"] = g.pressure_allowed.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    g["pressured"] = g.time_to_pressure.notna().astype("float64")  # DL generated pressure
    g["quick_pressure"] = g.quick_pressure.fillna(0.0)
    g["side_game"] = np.where(g.lat_10 < 0, "L", "R")
    num = GAME_METRICS + ["pressure_allowed", "peak_pressure_probability_allowed", "pressured",
                          "quick_pressure", "player_get_off", "sack", "sack_allowed", "dropback_duration",
                          "pass_rushers_encountered", "time_to_pressure", "lat_10"]
    for c in num:
        g[c] = pd.to_numeric(g[c], errors="coerce").astype("float64")
    return g


# ----------------------------------------------------------------------------
# per-player aggregation and reliability
# ----------------------------------------------------------------------------
def player_means(g: pd.DataFrame, by: list[str], metrics: list[str], min_plays: int = 50) -> pd.DataFrame:
    agg = g.groupby(by)[metrics].mean()
    agg["n_plays"] = g.groupby(by).size()
    return agg[agg.n_plays >= min_plays].reset_index()


def split_half_reliability(g: pd.DataFrame, by: list[str], metrics: list[str], min_plays: int = 50) -> pd.DataFrame:
    """Odd vs even weeks, Spearman-Brown corrected, per metric."""
    odd = g[g.week % 2 == 1]; even = g[g.week % 2 == 0]
    a = player_means(odd, by, metrics, min_plays // 2).set_index(by)
    b = player_means(even, by, metrics, min_plays // 2).set_index(by)
    common = a.index.intersection(b.index)
    rows = []
    for m in metrics:
        x, y = a.loc[common, m].astype("float64"), b.loc[common, m].astype("float64")
        ok = x.notna() & y.notna()
        r = np.corrcoef(x[ok].values, y[ok].values)[0, 1] if ok.sum() > 5 else np.nan
        rows.append({"metric": m, "n": int(ok.sum()), "r_half": r, "r_sb": 2 * r / (1 + r) if r == r else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# correlation with uncertainty / confounder control
# ----------------------------------------------------------------------------
def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r):
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999, 0.999)); se = 1 / np.sqrt(n - 3)
    from scipy.stats import norm
    q = norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - q * se)), float(np.tanh(z + q * se)))


def residualize(df: pd.DataFrame, cols: list[str], controls: list[str]) -> pd.DataFrame:
    """OLS-residualize each of `cols` on `controls` (with intercept)."""
    out = df.copy()
    X = np.column_stack([np.ones(len(df))] + [df[c].astype(float).values for c in controls])
    for c in cols:
        y = df[c].astype(float).values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[c] = y - X @ beta
    return out


def corr_table(df: pd.DataFrame, xs: list[str], ys: list[str], controls: list[str] | None = None,
               n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Pearson r for every x-y pair, bootstrap 95% CI over rows, optional
    partial correlation (both sides residualized on controls)."""
    rng = np.random.default_rng(seed)
    rows = []
    for x in xs:
        for y in ys:
            ctrl = [c for c in (controls or []) if c not in (x, y)]  # never control for x or y itself
            cols = list(dict.fromkeys([x, y] + ctrl))
            d = df[cols].dropna()
            n = len(d)
            if n < 8:
                rows.append({"x": x, "y": y, "n": n, "r": np.nan, "lo": np.nan, "hi": np.nan}); continue
            if ctrl:
                d = residualize(d, [x, y], ctrl)
            xv, yv = d[x].values, d[y].values
            r = np.corrcoef(xv, yv)[0, 1]
            idx = rng.integers(0, n, size=(n_boot, n))
            bx, by = xv[idx], yv[idx]
            bx = bx - bx.mean(1, keepdims=True); by = by - by.mean(1, keepdims=True)
            br = (bx * by).sum(1) / np.sqrt((bx ** 2).sum(1) * (by ** 2).sum(1))
            lo, hi = np.nanpercentile(br, [2.5, 97.5])
            rows.append({"x": x, "y": y, "n": n, "r": r, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def zscore_within(df: pd.DataFrame, cols: list[str], by: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        g = out.groupby(by)[c]
        out[c] = (out[c] - g.transform("mean")) / g.transform("std")
    return out
