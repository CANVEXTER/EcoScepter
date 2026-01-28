import numpy as np

def rosins_threshold(score: np.ndarray) -> float:
    """
    Implements Rosin's Unimodal Thresholding (Corner Detection).
    Ideal for change detection (Peak at 0, long tail for change).
    """
    # Filter for valid, positive data (the tail)
    data = score[np.isfinite(score)]
    data = data[data > 0] 
    
    if data.size < 100:
        return 0.3
        
    # 1. Compute Histogram
    hist, bin_edges = np.histogram(data, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 2. Find Peak (Mode) - usually near 0
    peak_idx = np.argmax(hist)
    peak_x = bin_centers[peak_idx]
    peak_y = hist[peak_idx]
    
    # 3. Find End of Tail (Last bin with significant data)
    try:
        # Find last bin with non-zero count
        last_idx = np.where(hist > 0)[0][-1]
    except IndexError:
        return 0.3
        
    last_x = bin_centers[last_idx]
    last_y = hist[last_idx]
    
    # 4. Define the Line from Peak to Tail
    # Line equation params: ax + by + c = 0
    a = peak_y - last_y
    b = last_x - peak_x
    c = (peak_x * last_y) - (last_x * peak_y)
    
    normalization = np.sqrt(a**2 + b**2)
    if normalization == 0:
        return 0.3

    # 5. Find Max Distance (The Elbow)
    # The point on the histogram curve furthest from the straight line
    distances = []
    # We only search strictly between the peak and the tail end
    candidates = range(peak_idx, last_idx)
    
    if not candidates:
        return 0.3
        
    for i in candidates:
        x0 = bin_centers[i]
        y0 = hist[i]
        d = np.abs(a*x0 + b*y0 + c) / normalization
        distances.append(d)
        
    # The index of max distance is the "corner"
    corner_local_idx = np.argmax(distances)
    corner_idx = candidates[corner_local_idx]
    
    return float(bin_centers[corner_idx])

def aggressiveness_to_threshold(
    score: np.ndarray,
    aggressiveness: float,
) -> float:
    """
    Hybrid approach: Uses Rosin's 'Real Insight' threshold as the anchor.
    Aggressiveness simply shifts slightly away from that scientific anchor.
    """
    # 1. Calculate the scientifically optimal "Elbow"
    rosin_t = rosins_threshold(score)
    
    # 2. Apply Aggressiveness as a modifier
    # Aggressiveness 0.5 = Exactly Rosin's threshold
    # Aggressiveness 1.0 = Lower (more sensitive, includes more change)
    # Aggressiveness 0.0 = Higher (more conservative, stricter)
    
    # We allow a +/- 50% shift from the optimal point
    modifier = 1.0 - (aggressiveness - 0.5) 
    
    final_t = rosin_t * modifier
    
    # Sanity clamps to prevent the threshold from being broken by extreme outliers
    return float(np.clip(final_t, 0.1, 0.6))

def apply_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    return score >= threshold

def auto_baseline_threshold(score: np.ndarray, percentile: float = 90.0) -> float:
    return float(np.nanpercentile(score, percentile))