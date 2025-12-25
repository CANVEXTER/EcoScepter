# scripts/thresholds.py
import numpy as np

def auto_baseline_threshold(
    score: np.ndarray,
    percentile: float = 90.0,
) -> float:
    """
    Derive a baseline threshold from the score distribution.

    Parameters
    ----------
    score : np.ndarray
        Continuous change score (H, W)
    percentile : float
        Upper percentile used as baseline

    Returns
    -------
    float
        Baseline threshold value
    """
    if not (0 < percentile < 100):
        raise ValueError("Percentile must be between 0 and 100")
    return float(np.nanpercentile(score, percentile))


def aggressiveness_to_threshold(
    score: np.ndarray,
    aggressiveness: float,
    base_percentile: float = 90.0,
    span: float = 15.0,
) -> float:
    """
    Map a user aggressiveness value [0,1] to a threshold.

    aggressiveness = 0   → conservative (higher threshold)
    aggressiveness = 1   → aggressive (lower threshold)

    The threshold is bounded within a percentile range to prevent extremes.
    """
    if not (0.0 <= aggressiveness <= 1.0):
        raise ValueError("Aggressiveness must be in [0, 1]")

    high_p = base_percentile + span / 2
    low_p  = base_percentile - span / 2

    target_p = high_p - aggressiveness * span
    return float(np.nanpercentile(score, target_p))


def apply_threshold(
    score: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Apply a scalar threshold to produce a boolean change mask.
    """
    return score >= threshold
