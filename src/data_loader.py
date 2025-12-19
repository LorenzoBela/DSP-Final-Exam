import pandas as pd
import numpy as np

def load_and_clean_data(filepath, start_date='2017-01-01'):
    """
    Loads traffic accident data, parses dates, handles missing values,
    and filters for the specified date range.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Convert dates
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    df['date'] = df['crash_date'].dt.date
    
    # Filter by date
    print(f"Original shape: {df.shape}")
    df = df[df['crash_date'] >= start_date]
    print(f"Filtered shape ({start_date}+): {df.shape}")
    
    # Fill missing values
    numeric_cols = ['injuries_total', 'injuries_fatal', 'injuries_incapacitating', 
                    'injuries_non_incapacitating', 'injuries_reported_not_evident', 'injuries_no_indication']
    # Check if cols exist before filling
    existing_cols = [c for c in numeric_cols if c in df.columns]
    df[existing_cols] = df[existing_cols].fillna(0)
    
    return df

def aggregate_daily_accidents(df):
    """
    Aggregates accident data into a daily time series.
    """
    daily_accidents = df.groupby('date').size().reset_index(name='accident_count')
    daily_accidents['date'] = pd.to_datetime(daily_accidents['date'])
    # Ensure continuous daily frequency
    daily_accidents = daily_accidents.set_index('date').asfreq('D', fill_value=0)
    
    return daily_accidents

def get_hourly_counts(df):
    return df.groupby('crash_hour').size()

def get_dow_counts(df):
    # Map raw numbers to something if needed, or just return counts
    return df.groupby('crash_day_of_week').size()
