import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt

def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    """
    Applies a time-domain Butterworth filter.
    """
    nyq = 0.5 * fs
    if isinstance(cutoff, tuple):
        normalized = (cutoff[0]/nyq, cutoff[1]/nyq)
    else:
        normalized = cutoff / nyq
    b, a = butter(order, normalized, btype=btype)
    return filtfilt(b, a, data)

def frequency_domain_filter(signal_data, fs, filter_type, cutoff):
    """
    Filters the signal in the frequency domain (ideal filter) using FFT/IFFT.
    
    Args:
        signal_data: Input signal array
        fs: Sampling frequency
        filter_type: 'lowpass', 'highpass', or 'bandpass'
        cutoff: Single value for low/high, tuple (low, high) for bandpass
        
    Returns:
        tuple: (filtered_signal_time_domain, filtered_fft_spectrum)
    """
    n = len(signal_data)
    fft_result = fft(signal_data)
    freqs = fftfreq(n, 1/fs)
    
    if filter_type == 'lowpass':
        mask = np.abs(freqs) <= cutoff
    elif filter_type == 'highpass':
        mask = np.abs(freqs) >= cutoff
    elif filter_type == 'bandpass':
        mask = (np.abs(freqs) >= cutoff[0]) & (np.abs(freqs) <= cutoff[1])
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    fft_filtered = fft_result * mask
    return np.real(ifft(fft_filtered)), fft_filtered
