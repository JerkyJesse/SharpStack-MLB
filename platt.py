"""Calibration helpers: Platt scaling + isotonic regression + season regression."""

import os
import json
import logging

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from scipy.interpolate import interp1d

from config import PLATT_SCALER_FILE, current_timestamp

ISOTONIC_FILE = "mlb_isotonic_scaler.json"


def fit_platt_scaler(probs, actuals):
    """
    Fit logistic regression on logit(raw_prob) -> actual outcome.
    Uses scipy.optimize.minimize (no sklearn needed).
    """
    raw = np.clip(np.array(probs, dtype=float), 1e-6, 1 - 1e-6)
    X   = logit(raw)
    y   = np.array(actuals, dtype=float)

    def neg_log_likelihood(params):
        coef, intercept = params
        p = expit(coef * X + intercept)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    result    = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="L-BFGS-B")
    coef, intercept = result.x
    scaler = {
        "intercept": float(intercept),
        "coef":      float(coef),
        "fitted_at": current_timestamp(),
        "n_samples": len(y),
    }
    logging.info("Platt scaler fitted: intercept=%.4f, coef=%.4f on %d samples",
                 scaler["intercept"], scaler["coef"], len(y))
    return scaler


def save_platt_scaler(scaler, filename=PLATT_SCALER_FILE):
    with open(filename, "w") as f:
        json.dump(scaler, f, indent=2)
    logging.info("Platt scaler saved -> %s", filename)


def load_platt_scaler(filename=PLATT_SCALER_FILE):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.debug("Failed to load Platt scaler: %s", e)
        return None


def apply_platt(raw_prob, scaler):
    """Map raw Elo probability through the fitted sigmoid."""
    if scaler is None:
        return raw_prob
    x = logit(np.clip(raw_prob, 1e-6, 1 - 1e-6))
    return float(expit(scaler["coef"] * x + scaler["intercept"]))


def _isotonic_regression(y, weights=None):
    """Pool Adjacent Violators Algorithm -- pure numpy, no sklearn."""
    n = len(y)
    if n == 0:
        return np.array([])
    if n == 1:
        return y.astype(float).copy()
    result = y.astype(float).copy()
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.array(weights, dtype=float)
    blocks = [[i] for i in range(n)]
    while True:
        merged = False
        new_blocks = [blocks[0]]
        for i in range(1, len(blocks)):
            prev_mean = np.average(result[new_blocks[-1]], weights=weights[new_blocks[-1]])
            curr_mean = np.average(result[blocks[i]], weights=weights[blocks[i]])
            if curr_mean < prev_mean:
                new_blocks[-1] = new_blocks[-1] + blocks[i]
                merged = True
            else:
                new_blocks.append(blocks[i])
        blocks = new_blocks
        if not merged:
            break
    for block in blocks:
        val = np.average(result[block], weights=weights[block])
        result[block] = val
    return result


def fit_isotonic_scaler(probs, actuals, n_bins=50):
    """
    Fit isotonic regression calibrator.
    Bins raw probabilities and fits monotone non-decreasing mapping.
    """
    raw = np.array(probs, dtype=float)
    y = np.array(actuals, dtype=float)
    order = np.argsort(raw)
    raw_sorted = raw[order]
    y_sorted = y[order]
    bin_size = max(1, len(raw_sorted) // n_bins)
    bin_probs = []
    bin_actuals = []
    for i in range(0, len(raw_sorted), bin_size):
        chunk_p = raw_sorted[i:i+bin_size]
        chunk_y = y_sorted[i:i+bin_size]
        bin_probs.append(float(np.mean(chunk_p)))
        bin_actuals.append(float(np.mean(chunk_y)))
    bin_probs = np.array(bin_probs)
    bin_actuals = np.array(bin_actuals)
    calibrated = _isotonic_regression(bin_actuals)
    calibrated = np.clip(calibrated, 0.001, 0.999)
    scaler = {
        "type": "isotonic",
        "x_points": bin_probs.tolist(),
        "y_points": calibrated.tolist(),
        "fitted_at": current_timestamp(),
        "n_samples": len(y),
        "n_bins": int(len(bin_probs)),
    }
    logging.info("Isotonic scaler fitted: %d bins on %d samples", len(bin_probs), len(y))
    return scaler


def apply_isotonic(raw_prob, scaler):
    """Map raw probability through fitted isotonic regression."""
    if scaler is None:
        return raw_prob
    x_pts = np.array(scaler["x_points"])
    y_pts = np.array(scaler["y_points"])
    f = interp1d(x_pts, y_pts, kind="linear", bounds_error=False,
                 fill_value=(y_pts[0], y_pts[-1]))
    return float(np.clip(f(raw_prob), 0.001, 0.999))


def save_isotonic_scaler(scaler, filename=ISOTONIC_FILE):
    with open(filename, "w") as f:
        json.dump(scaler, f, indent=2)
    logging.info("Isotonic scaler saved -> %s", filename)


def load_isotonic_scaler(filename=ISOTONIC_FILE):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.debug("Failed to load isotonic scaler: %s", e)
        return None


def regress_ratings_to_mean(ratings, factor=0.33):
    """Pull every team 33% toward league mean at season boundaries."""
    if not ratings:
        return ratings
    mean = float(np.mean(list(ratings.values())))
    for team in ratings:
        ratings[team] = ratings[team] + factor * (mean - ratings[team])
    logging.info("Season regression applied (factor=%.0f%%), mean=%.1f", factor * 100, mean)
    return ratings


# -- Beta Calibration --

BETA_SCALER_FILE = "mlb_beta_scaler.json"


def fit_beta_scaler(probs, actuals):
    """3-parameter beta calibration for asymmetric miscalibration.
    logit(p_cal) = c + a*log(p) - b*log(1-p)."""
    raw = np.clip(np.array(probs, dtype=float), 1e-6, 1 - 1e-6)
    y = np.array(actuals, dtype=float)

    def neg_ll(params):
        a, b, c = params
        logit_cal = c + a * np.log(raw) - b * np.log(1 - raw)
        p_cal = np.clip(expit(logit_cal), 1e-12, 1 - 1e-12)
        return -float(np.sum(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal)))

    result = minimize(neg_ll, x0=[1.0, 1.0, 0.0], method="L-BFGS-B",
                      bounds=[(0.01, 10.0), (0.01, 10.0), (-5.0, 5.0)])
    a, b, c = result.x
    scaler = {"type": "beta", "a": float(a), "b": float(b), "c": float(c),
              "fitted_at": current_timestamp(), "n_samples": len(y)}
    logging.info("Beta scaler fitted: a=%.4f b=%.4f c=%.4f on %d samples", a, b, c, len(y))
    return scaler


def apply_beta(raw_prob, scaler):
    """Map raw probability through fitted beta calibration."""
    if scaler is None:
        return raw_prob
    p = np.clip(raw_prob, 1e-6, 1 - 1e-6)
    logit_cal = scaler["c"] + scaler["a"] * np.log(p) - scaler["b"] * np.log(1 - p)
    return float(np.clip(expit(logit_cal), 0.001, 0.999))


def save_beta_scaler(scaler, filename=BETA_SCALER_FILE):
    with open(filename, "w") as f:
        json.dump(scaler, f, indent=2)
    logging.info("Beta scaler saved -> %s", filename)


def load_beta_scaler(filename=BETA_SCALER_FILE):
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.debug("Failed to load beta scaler: %s", e)
        return None
