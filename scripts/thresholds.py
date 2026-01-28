import numpy as np

def aggressiveness_to_threshold(
    score: np.ndarray,
    aggressiveness: float,
) -> float:
    """
    Determines a cutoff threshold for the change score.
    
    Aggressiveness 0.0 (Conservative) -> Higher Threshold
    Aggressiveness 1.0 (Sensitive)    -> Lower Threshold
    """
    if not (0.0 <= aggressiveness <= 1.0):
        aggressiveness = 0.5
    
    valid_scores = score[np.isfinite(score)]
    if valid_scores.size == 0:
        return 0.5 
        
    # 1. Statistical Bounds (Relative to image content)
    p_high = np.percentile(valid_scores, 99) 
    p_low  = np.percentile(valid_scores, 85)
    
    # 2. Physical Bounds (Hard limits)
    # Score > 0.4 is almost certainly real degradation
    # Score < 0.1 is usually just noise
    phy_high = 0.4
    phy_low = 0.1
    
    # Interpolate based on aggressiveness
    # Conservative (0.0) -> wants High Threshold (p_high or phy_high)
    target_p = p_high - (aggressiveness * (p_high - p_low))
    target_phy = phy_high - (aggressiveness * (phy_high - phy_low))
    
    # Take the MAXIMUM to avoid flagging noise in static images
    final_threshold = max(target_p, target_phy)
    
    return float(final_threshold)

def apply_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    return score >= threshold

def auto_baseline_threshold(score: np.ndarray, percentile: float = 90.0) -> float:
    return float(np.nanpercentile(score, percentile))