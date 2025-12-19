import numpy as np
from scipy.fft import fft, fftfreq

def compute_fft(signal_data, sampling_rate=1):
    """
    Computes the FFT of a real-valued signal.
    
    Args:
        signal_data (array-like): The time-series signal.
        sampling_rate (float): Sampling rate (e.g., 1 for daily data).
        
    Returns:
        tuple: (positive_frequencies, magnitudes, full_fft_result)
    """
    n = len(signal_data)
    fft_result = fft(signal_data)
    freqs = fftfreq(n, 1/sampling_rate)
    
    # Keep only positive frequencies
    positive_idx = freqs >= 0
    freqs_pos = freqs[positive_idx]
    # Magnitude: |FFT| * 2 / N
    magnitude = np.abs(fft_result[positive_idx]) * 2 / n
    
    return freqs_pos, magnitude, fft_result

def get_dominant_frequencies(freqs, magnitude, top_n=10):
    """Identifies top N dominant frequencies."""
    # Exclude DC component (0 Hz) usually at index 0
    valid_idx = np.where(freqs > 0)[0]
    
    sorted_indices = np.argsort(magnitude[valid_idx])[-top_n:][::-1]
    top_indices = valid_idx[sorted_indices]
    
    results = []
    for idx in top_indices:
        results.append({
            'frequency': freqs[idx],
            'period': 1/freqs[idx],
            'magnitude': magnitude[idx]
        })
    return results
