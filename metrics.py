"""Scoring metrics: log loss, Brier score, calibration table."""

import numpy as np
import pandas as pd


def log_loss_binary(y_true, p_pred, eps=1e-12):
    p = np.clip(np.array(p_pred, dtype=float), eps, 1 - eps)
    y = np.array(y_true, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier_score_binary(y_true, p_pred):
    y = np.array(y_true, dtype=float)
    p = np.array(p_pred, dtype=float)
    return float(((p - y) ** 2).mean())


def calibration_table(probs, actuals, bins=10):
    df = pd.DataFrame({"prob": probs, "actual": actuals})
    df["bucket"] = pd.cut(df["prob"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return df.groupby("bucket", observed=False).agg(
        n=("actual","size"), avg_prob=("prob","mean"), actual_rate=("actual","mean")
    ).reset_index()


def ece_score(probs, actuals, bins=10):
    """Expected Calibration Error: weighted average of per-bin calibration gap.
    ECE < 0.03 is excellent; > 0.08 indicates miscalibration."""
    p = np.array(probs, dtype=float)
    y = np.array(actuals, dtype=float)
    n = len(p)
    if n == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        n_b = mask.sum()
        if n_b == 0:
            continue
        ece += (n_b / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def mce_score(probs, actuals, bins=10):
    """Maximum Calibration Error: worst single-bin calibration gap."""
    p = np.array(probs, dtype=float)
    y = np.array(actuals, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    mce = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(float(p[mask].mean()) - float(y[mask].mean())))
    return float(mce)


def brier_skill_score(probs, actuals, reference_probs=None):
    """BSS = 1 - (Brier_model / Brier_reference).  BSS > 0 means model beats reference.
    Default reference: constant 50% (no-skill baseline).
    Also computes vs home-win-rate baseline."""
    brier_model = brier_score_binary(actuals, probs)
    if reference_probs is None:
        reference_probs = [0.5] * len(actuals)
    brier_ref = brier_score_binary(actuals, reference_probs)
    if brier_ref < 1e-12:
        return 0.0
    return float(1.0 - brier_model / brier_ref)


def conformal_prediction_set(probs, actuals, alpha=0.10):
    """Distribution-free conformal prediction intervals with guaranteed coverage.
    alpha=0.10 -> 90% target coverage.  Returns coverage stats."""
    p = np.array(probs, dtype=float)
    y = np.array(actuals, dtype=float)
    n = len(p)
    if n < 30:
        return None
    # Nonconformity score = 1 - predicted_prob_of_actual_outcome
    scores = np.where(y == 1, 1 - p, p)
    # Expanding-window conformal: use past scores to set threshold
    cal_start = 20
    set_sizes = []
    covered = 0
    for i in range(cal_start, n):
        q = np.quantile(scores[:i], 1 - alpha)
        home_nc = 1 - p[i]  # nonconformity if home wins
        away_nc = p[i]       # nonconformity if away wins
        pred_set = []
        if home_nc <= q:
            pred_set.append(1)
        if away_nc <= q:
            pred_set.append(0)
        set_sizes.append(len(pred_set))
        if int(y[i]) in pred_set:
            covered += 1
    total = n - cal_start
    if total == 0:
        return None
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "actual_coverage": covered / total,
        "avg_set_size": float(np.mean(set_sizes)),
        "singleton_pct": sum(1 for s in set_sizes if s == 1) / total * 100,
        "empty_pct": sum(1 for s in set_sizes if s == 0) / total * 100,
        "both_pct": sum(1 for s in set_sizes if s == 2) / total * 100,
        "n_evaluated": total,
    }
