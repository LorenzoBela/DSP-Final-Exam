import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Configure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

def ensure_output_dir(output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def plot_time_series(dates, counts, title, filename):
    plt.figure(figsize=(14, 6))
    plt.plot(dates, counts, linewidth=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_bar_chart(x, y, title, xlabel, ylabel, filename, color='steelblue', xticks=None, xticklabels=None, orientation='v'):
    plt.figure(figsize=(12, 6) if orientation=='v' else (10, 8))
    
    if orientation == 'v':
        plt.bar(x, y, color=color, edgecolor='navy')
        if xticks:
            plt.xticks(xticks, xticklabels if xticklabels else xticks)
    else:
        plt.barh(x, y, color=color, edgecolor='navy')
        # Invert y-axis for horizontal bars usually so top item is at top
        plt.gca().invert_yaxis()
        
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='x' if orientation=='h' else 'y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_fft_spectrum(freqs, magnitude, title, filename, top_n=5):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot spectrum (skip DC at index 0 usually, or assume freqs[0]=0)
    # We'll plot from index 1 to avoid the massive DC component scaling issues
    start_idx = 1 if freqs[0] == 0 else 0
    
    ax.plot(freqs[start_idx:], magnitude[start_idx:], 'b-', linewidth=0.8)
    ax.fill_between(freqs[start_idx:], magnitude[start_idx:], alpha=0.3)
    
    # Find peaks for annotation
    mag_view = magnitude[start_idx:]
    freq_view = freqs[start_idx:]
    
    # Simple peak finding on the view
    peak_indices = np.argsort(mag_view)[-top_n:]
    
    for i in peak_indices:
        pk_freq = freq_view[i]
        pk_mag = mag_view[i]
        
        # Threshold check
        if pk_mag > np.mean(mag_view) * 2:
            period = 1/pk_freq if pk_freq > 0 else 0
            ax.annotate(f'{period:.1f}d',
                       xy=(pk_freq, pk_mag),
                       xytext=(pk_freq, pk_mag * 1.15),
                       ha='center', fontsize=10,
                       arrowprops=dict(arrowstyle='->', color='red'))
            ax.plot(pk_freq, pk_mag, 'ro', markersize=6)
            
    ax.set_title(f'Frequency Spectrum - {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frequency (cycles/day)')
    ax.set_ylabel('Magnitude')
    ax.set_xlim(0, 0.2) # Focus on low freq
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_filter_comparison(time, original, filtered, title, filename, label_original='Original', label_filtered='Filtered'):
    plt.figure(figsize=(14, 6))
    plt.plot(time, original, 'b-', linewidth=0.3, alpha=0.5, label=label_original)
    plt.plot(time, filtered, 'g-', linewidth=1.5, label=label_filtered)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df, title, filename):
    # Matches the guide.md example code block structure
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(
        correlation_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm',
        linewidths=.5, 
        center=0, # Keeping center=0 is good practice for correlations even if not in snippet
        square=True,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title('Correlation Matrix of Time Series Parameters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
