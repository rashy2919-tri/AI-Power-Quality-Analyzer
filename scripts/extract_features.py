import numpy as np
import pandas as pd
import os

# ==========================================
# 1. Configuration
# ==========================================
DATASET_DIR = 'dataset'
CLASSES = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Transient']
OUTPUT_FILE = 'features.csv'

# ==========================================
# 2. Feature Extraction Functions
# ==========================================

def calculate_rms(signal):
    """Calculate Root Mean Square (RMS)"""
    return np.sqrt(np.mean(signal**2))

def calculate_thd(signal, fs):
    """Calculate Total Harmonic Distortion (THD)"""
    N = len(signal)
    # Perform Fast Fourier Transform
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, 1/fs)
    
    # Get positive frequencies only
    idx = np.where(xf > 0)
    xf = xf[idx]
    yf = np.abs(yf[idx]) / N
    
    # Find the fundamental frequency (closest to 50 Hz)
    fundamental_idx = np.argmin(np.abs(xf - 50))
    fundamental_mag = yf[fundamental_idx]
    
    # Calculate harmonics (2nd to 15th)
    harmonic_mags = []
    for h in range(2, 16):
        target_freq = 50 * h
        h_idx = np.argmin(np.abs(xf - target_freq))
        harmonic_mags.append(yf[h_idx])
    
    # THD Formula
    if fundamental_mag == 0:
        return 0
    thd = np.sqrt(np.sum(np.array(harmonic_mags)**2)) / fundamental_mag
    return thd * 100 # Return as percentage

def calculate_frequency(signal, fs):
    """Calculate frequency using FFT (robust against noise)"""
    N = len(signal)
    yf = np.fft.fft(signal)
    xf = np.fft.fftfreq(N, 1/fs)
    
    # Get positive frequencies only
    idx = np.where(xf > 0)
    xf = xf[idx]
    yf = np.abs(yf[idx]) / N
    
    # Find the peak frequency (the fundamental frequency)
    # Ignore the first few points (DC offset/noise)
    peak_idx = np.argmax(yf[5:]) + 5 
    return xf[peak_idx]
# ==========================================
# 3. Main Processing Loop
# ==========================================
def main():
    print("Starting feature extraction...")
    data = []
    
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        files = [f for f in os.listdir(cls_dir) if f.endswith('.csv')]
        
        print(f"Processing {len(files)} files for class: {cls}...")
        
        for filename in files:
            filepath = os.path.join(cls_dir, filename)
            df = pd.read_csv(filepath)
            
            # Extract signals
            v = df['voltage'].values
            i = df['current'].values
            t = df['time'].values
            fs = 1 / (t[1] - t[0]) # Calculate sampling frequency from time array
            
            # Extract Features
            v_rms = calculate_rms(v)
            i_rms = calculate_rms(i)
            thd_v = calculate_thd(v, fs)
            freq = calculate_frequency(v, fs)
            
            # Append to our list
            data.append({
                'V_rms': v_rms,
                'I_rms': i_rms,
                'THD_V': thd_v,
                'Frequency': freq,
                'Label': cls
            })
            
    # Save to CSV
    features_df = pd.DataFrame(data)
    features_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nSuccess! Extracted features saved to '{OUTPUT_FILE}'")
    print(f"Total rows in dataset: {len(features_df)}")
    print("\nSample of extracted features:")
    print(features_df.head())

if __name__ == "__main__":
    main()