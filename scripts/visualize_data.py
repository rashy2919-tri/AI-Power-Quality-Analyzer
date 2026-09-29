import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# ==========================================
# 1. Configuration
# ==========================================
DATASET_DIR = 'dataset'
CLASSES = ['Normal', 'Sag', 'Swell', 'Harmonic', 'Transient']

# ==========================================
# 2. Plotting Logic
# ==========================================
def plot_samples():
    # Create a figure with 5 subplots (one for each class), stacked vertically
    fig, axes = plt.subplots(nrows=len(CLASSES), ncols=1, figsize=(10, 12))
    fig.suptitle('Power Quality Disturbance Classes', fontsize=16, fontweight='bold')

    for i, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATASET_DIR, cls)
        
        # Check if folder exists
        if not os.path.exists(cls_dir):
            print(f"Warning: Folder {cls_dir} not found. Did you run the dataset generator?")
            continue
            
        # Get a list of all CSV files in this class folder
        files = [f for f in os.listdir(cls_dir) if f.endswith('.csv')]
        
        if not files:
            print(f"Warning: No CSV files found in {cls_dir}")
            continue
            
        # Pick a random file to plot
        random_file = random.choice(files)
        filepath = os.path.join(cls_dir, random_file)
        
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # Plot the Voltage column
        ax = axes[i]
        ax.plot(df['time'] * 1000, df['voltage'], color='blue', linewidth=1.5) # Convert time to milliseconds
        
        # Formatting the subplot
        ax.set_title(f'Class: {cls} (File: {random_file})', fontsize=12)
        ax.set_ylabel('Voltage (V)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only put the x-label on the bottom plot
        if i == len(CLASSES) - 1:
            ax.set_xlabel('Time (ms)', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for the main title
    
    # Save the plot as an image
    output_image = 'dataset_validation_plot.png'
    plt.savefig(output_image, dpi=300)
    print(f"\nSuccess! Plot saved as '{output_image}' in your current folder.")
    
    # Show the plot on screen
    plt.show()

if __name__ == "__main__":
    print("Generating validation plots...")
    plot_samples()