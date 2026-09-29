import numpy as np
import pandas as pd
import os

# ==========================================
# 1. Configuration
# ==========================================
fs = 20000          # Sampling frequency (20 kHz) - standard for power quality
duration = 0.2      # Duration of each sample (200 ms) - 10 cycles at 50Hz
t = np.arange(0, duration, 1/fs)
f0 = 50             # Nominal frequency (50 Hz)
V_rms = 230         # Nominal RMS voltage
V_peak = V_rms * np.sqrt(2)

# Number of samples to generate per class
# START SMALL FOR TESTING. Later, change this to 500 or 1000.
N_SAMPLES = 50 
OUTPUT_DIR = 'dataset'

# ==========================================
# 2. Disturbance Generation Functions
# ==========================================

def add_noise(signal, noise_level=0.01):
    """Adds Gaussian noise to the signal to simulate real-world conditions"""
    noise = np.random.normal(0, noise_level * V_peak, len(signal))
    return signal + noise

def generate_normal(t):
    """Normal sinusoidal voltage"""
    return V_peak * np.sin(2 * np.pi * f0 * t)

def generate_sag(t, depth, t_start, duration):
    """Voltage Sag (Dip)"""
    v = V_peak * np.sin(2 * np.pi * f0 * t)
    mask = (t >= t_start) & (t <= t_start + duration)
    v[mask] *= (1 - depth)
    return v

def generate_swell(t, factor, t_start, duration):
    """Voltage Swell"""
    v = V_peak * np.sin(2 * np.pi * f0 * t)
    mask = (t >= t_start) & (t <= t_start + duration)
    v[mask] *= factor
    return v

def generate_harmonic(t, harmonics):
    """Voltage with Harmonic Distortion"""
    v = V_peak * np.sin(2 * np.pi * f0 * t)
    for order, mag in harmonics.items():
        v += V_peak * mag * np.sin(2 * np.pi * f0 * order * t)
    return v

def generate_transient(t, t_start, freq_osc, decay):
    """Oscillatory Transient"""
    v = V_peak * np.sin(2 * np.pi * f0 * t)
    mask = t >= t_start
    transient = 0.3 * V_peak * np.exp(-decay * (t - t_start)) * \
                np.sin(2 * np.pi * freq_osc * (t - t_start))
    v[mask] += transient[mask]
    return v

def calculate_current(v, R=50):
    """Calculate current assuming a purely resistive load (R=50 ohms)"""
    return v / R

# ==========================================
# 3. Dataset Generation Loop
# ==========================================
def main():
    print("Starting dataset generation...")
    classes = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Transient']
    
    for cls in classes:
        cls_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print(f"Generating {N_SAMPLES} samples for class: {cls}...")
        
        for i in range(N_SAMPLES):
            # Randomize parameters to make the AI model robust
            if cls == 'Normal':
                v = generate_normal(t)
                
            elif cls == 'Sag':
                depth = np.random.uniform(0.1, 0.9)
                t_start = np.random.uniform(0.02, 0.1)
                dur = np.random.uniform(0.02, 0.15)
                v = generate_sag(t, depth, t_start, dur)
                
            elif cls == 'Swell':
                factor = np.random.uniform(1.1, 1.5)
                t_start = np.random.uniform(0.02, 0.1)
                dur = np.random.uniform(0.02, 0.15)
                v = generate_swell(t, factor, t_start, dur)
                
            elif cls == 'Harmonic':
                h_orders = [3, 5, 7] # 3rd, 5th, and 7th harmonics
                harmonics = {order: np.random.uniform(0.05, 0.15) for order in h_orders}
                v = generate_harmonic(t, harmonics)
                
            elif cls == 'Transient':
                t_start = np.random.uniform(0.02, 0.1)
                freq_osc = np.random.uniform(500, 2000)
                decay = np.random.uniform(20, 100)
                v = generate_transient(t, t_start, freq_osc, decay)
            
            # Add noise to all signals
            v = add_noise(v, noise_level=0.01)
            
            # Calculate current
            i_signal = calculate_current(v)
            
            # Save to CSV
            df = pd.DataFrame({
                'time': t,
                'voltage': v,
                'current': i_signal
            })
            
            filename = os.path.join(cls_dir, f'{cls}_{i:04d}.csv')
            df.to_csv(filename, index=False)
            
    print(f"\nSuccess! Dataset generated in the '{OUTPUT_DIR}' folder.")
    print(f"Total files: {N_SAMPLES * len(classes)}")

if __name__ == "__main__":
    main()