import os
import pandas as pd
from src.data_loader import load_and_clean_data, aggregate_daily_accidents, get_hourly_counts, get_dow_counts
from src.fft_processor import compute_fft, get_dominant_frequencies
from src.signal_filter import frequency_domain_filter
from src.visualizer import (ensure_output_dir, plot_time_series, plot_bar_chart, 
                           plot_fft_spectrum, plot_filter_comparison, plot_correlation_matrix)

def main():
    # Setup
    DATA_PATH = 'data/traffic_accidents.csv'
    OUTPUT_DIR = ensure_output_dir('outputs_py') # Use a separate folder to distinguish from notebook items
    
    print("=== DSP Traffic Analysis Pipeline ===")
    
    # 1. Load Data
    df = load_and_clean_data(DATA_PATH, start_date='2017-01-01')
    
    # 2. Preprocessing
    print("\n--- Preprocessing ---")
    daily_accidents = aggregate_daily_accidents(df)
    hourly_counts = get_hourly_counts(df)
    dow_counts = get_dow_counts(df)
    
    # Save processed data
    daily_accidents.to_csv(os.path.join('data', 'cleaned_daily_accidents_py.csv'))
    
    # 3. Initial Visualizations
    print("\n--- Generating Time Series Visualizations ---")
    plot_time_series(daily_accidents.index, daily_accidents['accident_count'],
                    'Daily Traffic Accidents (2017-2025)',
                    os.path.join(OUTPUT_DIR, '1_daily_time_series.png'))
    
    plot_bar_chart(hourly_counts.index, hourly_counts.values,
                  'Accidents by Hour of Day', 'Hour', 'Count',
                  os.path.join(OUTPUT_DIR, '2_hourly_distribution.png'),
                  xticks=range(24))
                  
    dow_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    # Ensure index 1-7 covers everything, get values safely
    dow_vals = [dow_counts.get(i, 0) for i in range(1, 8)]
    plot_bar_chart(range(1, 8), dow_vals,
                  'Accidents by Day of Week', 'Day', 'Count',
                  os.path.join(OUTPUT_DIR, '3_dow_distribution.png'),
                  color='coral', xticks=range(1, 8), xticklabels=dow_labels)

    # 4. FFT Analysis
    print("\n--- Running FFT Analysis ---")
    signal = daily_accidents['accident_count'].values
    freqs, magnitudes, _ = compute_fft(signal, sampling_rate=1)
    
    plot_fft_spectrum(freqs, magnitudes, 'Daily Traffic Accidents',
                     os.path.join(OUTPUT_DIR, '4_fft_spectrum.png'))
                     
    dom_freqs = get_dominant_frequencies(freqs, magnitudes)
    print("Top Dominant Frequencies:")
    for item in dom_freqs[:5]:
        print(f"  Period: {item['period']:.2f} days | Mag: {item['magnitude']:.2f}")

    # 5. Filtering
    print("\n--- Applying Spectral Filters ---")
    
    # Original Signal Plot (Missing item 1)
    plot_time_series(daily_accidents.index, signal, 
                    'Original Signal (Daily Accidents)',
                    os.path.join(OUTPUT_DIR, '5a_signal_original.png'))

    # Low pass
    lp_cutoff = 0.1
    filtered_lp, _ = frequency_domain_filter(signal, 1, 'lowpass', lp_cutoff)
    
    # Comparison Plot (Missing item 2)
    plot_filter_comparison(daily_accidents.index, signal, filtered_lp,
                          f'Comparison: Original vs Low-Pass (Cutoff={lp_cutoff})',
                          os.path.join(OUTPUT_DIR, '5b_signal_comparison.png'))

    plot_filter_comparison(daily_accidents.index, signal, filtered_lp,
                          f'Low-Pass Filtered Signal',
                          os.path.join(OUTPUT_DIR, '5c_lowpass_filter.png'),
                          label_original=None, label_filtered='Low-Pass Signal')
                          
    # Band pass (Weekly: ~0.143, use 0.12-0.16)
    bp_cutoff = (0.12, 0.16)
    filtered_bp, _ = frequency_domain_filter(signal, 1, 'bandpass', bp_cutoff)
    # For bandpass, we might want to plot just the component vs time, or compare
    # Here illustrating just the component
    plot_filter_comparison(daily_accidents.index, signal, filtered_bp,
                          'Band-Pass Filter (Weekly Cycle)',
                          os.path.join(OUTPUT_DIR, '6_bandpass_filter.png'),
                          label_filtered='Weekly Component')

    # 6. Correlation Analysis
    print("\n--- Running Correlation Analysis ---")
    num_cols = ['crash_hour', 'crash_day_of_week', 'crash_month', 'num_units',
              'injuries_total', 'injuries_fatal', 'injuries_incapacitating',
              'injuries_non_incapacitating']
    plot_correlation_matrix(df[num_cols], 'Correlation Matrix',
                           os.path.join(OUTPUT_DIR, '7_correlation_matrix.png'))
                           
    # Weather/Lighting
    weather = df['weather_condition'].value_counts().head(10)
    plot_bar_chart(weather.index, weather.values, 'Top 10 Weather Conditions',
                  'Count', 'Condition',
                  os.path.join(OUTPUT_DIR, '8_weather_conditions.png'),
                  orientation='h')
                  
    lighting = df['lighting_condition'].value_counts().head(8)
    plot_bar_chart(lighting.index, lighting.values, 'Lighting Conditions',
                  'Count', 'Condition',
                  os.path.join(OUTPUT_DIR, '9_lighting_conditions.png'),
                  orientation='h', color='coral')

    print(f"\n✅ Analysis complete! Outputs saved in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()
