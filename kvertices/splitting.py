import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from typing import Union
# Add other necessary imports like scipy.stats for KDE later

def split_1d_median(eta: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Splits the data based on the median of the 1D embedding eta.

    Parameters
    ----------
    eta : np.ndarray, shape (n,)
        The 1D embedding values.

    Returns
    -------
    indices1 : np.ndarray
        Local indices belonging to the first cluster.
    indices2 : np.ndarray
        Local indices belonging to the second cluster.
    """
    if eta.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
        
    median_val = np.median(eta)
    # Handle case where all values are the same
    if np.all(eta == median_val):
         # Split arbitrarily, e.g., first half and second half
         mid_point = eta.size // 2
         indices1 = np.arange(mid_point)
         indices2 = np.arange(mid_point, eta.size)
         return indices1, indices2

    # Split based on median
    # Points exactly equal to median can go to either side, 
    # here they go to indices1 (<= median)
    indices1 = np.where(eta <= median_val)[0]
    indices2 = np.where(eta > median_val)[0]
    
    # Ensure non-empty clusters if possible
    if len(indices1) == 0:
        # Move the smallest value from indices2 to indices1
        min_idx_in_2 = indices2[np.argmin(eta[indices2])]
        indices1 = np.array([min_idx_in_2])
        indices2 = np.setdiff1d(indices2, indices1, assume_unique=True)
    elif len(indices2) == 0:
        # Move the largest value from indices1 to indices2
        max_idx_in_1 = indices1[np.argmax(eta[indices1])]
        indices2 = np.array([max_idx_in_1])
        indices1 = np.setdiff1d(indices1, indices2, assume_unique=True)
        
    return indices1, indices2

def split_1d_kmeans(eta: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Splits the data using the optimal 1D k-means threshold (k=2).
    Finds the threshold that minimizes the sum of within-cluster variances.
    Uses an efficient O(n log n) or O(n) approach after sorting.

    Parameters
    ----------
    eta : np.ndarray, shape (n,)
        The 1D embedding values.

    Returns
    -------
    indices1 : np.ndarray
        Local indices belonging to the first cluster.
    indices2 : np.ndarray
        Local indices belonging to the second cluster.
    """
    n = eta.size
    if n <= 1:
        return np.arange(n), np.array([], dtype=int)

    # Sort eta and get original indices
    sorted_indices = np.argsort(eta)
    eta_sorted = eta[sorted_indices]

    # Handle case where all values are the same
    if np.abs(eta_sorted[0] - eta_sorted[-1]) < 1e-9:
        mid_point = n // 2
        indices1 = sorted_indices[:mid_point]
        indices2 = sorted_indices[mid_point:]
        return indices1, indices2
        
    # Calculate prefix sums for efficient variance calculation
    # cumsum includes the current element, so pad with 0 at the start
    s1 = np.concatenate(([0], np.cumsum(eta_sorted)))
    s2 = np.concatenate(([0], np.cumsum(eta_sorted**2)))

    min_sse = np.inf
    best_split_index = -1 # Index *after* the split point in the sorted array

    # Iterate through all possible split points (after index i-1, before index i)
    for i in range(1, n):
        # Cluster 1: indices 0 to i-1 (size i)
        sse1 = s2[i] - (s1[i]**2) / i
        
        # Cluster 2: indices i to n-1 (size n-i)
        sum1_c2 = s1[n] - s1[i]
        sum2_c2 = s2[n] - s2[i]
        count_c2 = n - i
        sse2 = sum2_c2 - (sum1_c2**2) / count_c2
        
        total_sse = sse1 + sse2
        
        if total_sse < min_sse:
            min_sse = total_sse
            best_split_index = i

    # Split the original indices based on the best split found
    indices1 = sorted_indices[:best_split_index]
    indices2 = sorted_indices[best_split_index:]

    return indices1, indices2

def split_1d_max_gap(eta: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Splits the data by finding the largest gap between sorted eta values.

    Parameters
    ----------
    eta : np.ndarray, shape (n,)
        The 1D embedding values.

    Returns
    -------
    indices1 : np.ndarray
        Local indices belonging to the first cluster (values <= threshold).
    indices2 : np.ndarray
        Local indices belonging to the second cluster (values > threshold).
    """
    n = eta.size
    if n <= 1:
        # Cannot split if 0 or 1 point
        return np.arange(n), np.array([], dtype=int)

    # Sort eta and get the original indices
    sorted_indices = np.argsort(eta)
    eta_sorted = eta[sorted_indices]

    # Calculate gaps between consecutive sorted points
    gaps = np.diff(eta_sorted)

    if not np.any(gaps > 1e-9): # Handle case where all values are effectively the same
         mid_point = n // 2
         indices1 = sorted_indices[:mid_point]
         indices2 = sorted_indices[mid_point:]
         return indices1, indices2
         
    # Find the index *before* the largest gap
    max_gap_index = np.argmax(gaps)

    # Split the *original* indices based on the position of the largest gap
    # Indices up to and including max_gap_index belong to the first cluster
    indices1 = sorted_indices[:max_gap_index + 1]
    indices2 = sorted_indices[max_gap_index + 1:]

    return indices1, indices2

def split_1d_kde(eta: np.ndarray, bandwidth: Union[float, str, None] = None, grid_size=512) -> (np.ndarray, np.ndarray):
    """
    Splits the data by finding a valley in the Kernel Density Estimate (KDE).
    Identifies the two main peaks and finds the minimum density between them.

    Parameters
    ----------
    eta : np.ndarray, shape (n,)
        The 1D embedding values.
    bandwidth : float or str, optional
        The bandwidth for the KDE. Can be a scalar, or methods like
        'scott', 'silverman'. If None, default for gaussian_kde is used ('scott').
    grid_size : int, optional
        Number of points to evaluate the KDE on.

    Returns
    -------
    indices1 : np.ndarray
        Local indices belonging to the first cluster.
    indices2 : np.ndarray
        Local indices belonging to the second cluster.
    """
    n = eta.size
    if n <= 1:
        return np.arange(n), np.array([], dtype=int)

    # Handle constant data - KDE is ill-defined or trivial
    if np.abs(np.min(eta) - np.max(eta)) < 1e-9:
        # Split arbitrarily like other methods
        mid_point = n // 2
        sorted_indices = np.argsort(eta) # Use sorted indices for consistency
        indices1 = sorted_indices[:mid_point]
        indices2 = sorted_indices[mid_point:]
        return indices1, indices2

    try:
        # Create KDE object
        # gaussian_kde requires at least 2 points if dataset has variance > 0
        if n < 2:
             # Should not happen due to initial check, but safeguard
             return np.arange(n), np.array([], dtype=int)
             
        # Explicitly handle cases where variance is zero after filtering etc.
        if np.std(eta) < 1e-9:
             mid_point = n // 2
             sorted_indices = np.argsort(eta)
             indices1 = sorted_indices[:mid_point]
             indices2 = sorted_indices[mid_point:]
             return indices1, indices2
             
        kde = gaussian_kde(eta, bw_method=bandwidth)

        # Evaluate KDE on a grid
        min_val, max_val = np.min(eta), np.max(eta)
        padding = 0.1 * (max_val - min_val) if (max_val - min_val) > 1e-6 else 0.1
        x_grid = np.linspace(np.min(eta) - padding, np.max(eta) + padding, grid_size)
        density = kde.evaluate(x_grid)

        # Find peaks in the density
        # Adjust height/distance parameters as needed for robustness
        peaks, properties = find_peaks(density, height=0)

        if len(peaks) < 2:
            # Not enough peaks to find a valley between them.
            # Fallback to median or max_gap?
            # print("Warning: KDE split found < 2 peaks. Falling back to median.")
            return split_1d_median(eta) # Fallback to median split

        # Find the two highest peaks
        peak_heights = properties['peak_heights']
        sorted_peak_indices = np.argsort(peak_heights)[::-1] # Indices of peaks, sorted by height desc
        peak1_idx = peaks[sorted_peak_indices[0]]
        peak2_idx = peaks[sorted_peak_indices[1]]

        # Ensure peak1 is the leftmost of the two highest peaks
        if peak1_idx > peak2_idx:
            peak1_idx, peak2_idx = peak2_idx, peak1_idx

        # Find the index of the minimum density *between* these two peaks
        valley_idx = peak1_idx + np.argmin(density[peak1_idx:peak2_idx + 1])

        # Threshold is the grid point at the valley index
        threshold = x_grid[valley_idx]

        # Split original indices based on the threshold
        indices1 = np.where(eta <= threshold)[0]
        indices2 = np.where(eta > threshold)[0]
        
        # Handle potential empty clusters after thresholding (though unlikely if valley exists)
        if len(indices1) == 0 or len(indices2) == 0:
             # print("Warning: KDE split resulted in an empty cluster. Falling back to median.")
             return split_1d_median(eta)

        return indices1, indices2

    except Exception as e:
        # Catch potential errors during KDE or peak finding
        print(f"Warning: Error during KDE split: {e}. Falling back to median.")
        return split_1d_median(eta) 