import pandas as pd
import numpy as np
from scipy.stats import skew

def extract_features(raw_df):
    df = raw_df.copy()
    window_size = 20
    
    # 1. Fitur Gerakan
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    
    # 2. Fitur Temporal (Rolling Window)
    df['hr_roll_mean'] = df['hr'].rolling(window=window_size, min_periods=1).mean()
    df['hr_roll_std'] = df['hr'].rolling(window=window_size, min_periods=1).std().fillna(0)
    
    df['eda_phasic_roll_mean'] = df['eda_phasic'].rolling(window=window_size, min_periods=1).mean()
    
    def calculate_skew(x):
        return skew(x) if len(x) > 2 else 0
        
    df['eda_phasic_roll_skew'] = df['eda_phasic'].rolling(window=window_size, min_periods=1).apply(calculate_skew, raw=True)
    
    # 3. Seleksi Fitur
    features_to_keep = [
        'eda_tonic', 'eda_phasic', 'hr_roll_mean', 'hr_roll_std', 
        'eda_phasic_roll_mean', 'eda_phasic_roll_skew', 'acc_mag', 'bvp', 'temp'
    ]
    
    # Ambil baris terakhir saja
    final_features = df[features_to_keep].iloc[[-1]] 
    
    return final_features