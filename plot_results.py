import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from math import pi

# Set a clean, professional style
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. LINE GRAPH (Time Comparison)
# ==========================================
def plot_line_filled(csv_file, title_suffix, output_filename):
    if not os.path.exists(csv_file):
        print(f"Skipping {csv_file} (File not found)")
        return

    df = pd.read_csv(csv_file)
    
    # Standardize column names
    if 'TotalSerialTime(s)' in df.columns:
        df = df.rename(columns={'TotalSerialTime(s)': 'SerialTime', 'TotalParallelTime(s)': 'ParallelTime'})

    filters = df['Filter']
    serial = df['SerialTime']
    parallel = df['ParallelTime']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Lines
    ax.plot(filters, serial, marker='o', linestyle='-', linewidth=2, color='#C44E52', label='Serial (Slow)')
    ax.plot(filters, parallel, marker='o', linestyle='-', linewidth=2, color='#4C72B0', label='Parallel (Fast)')

    # Fill the area between them to show "Time Saved"
    ax.fill_between(filters, serial, parallel, color='#55A868', alpha=0.2, label='Time Saved')

    # Annotate points with values
    for i, txt in enumerate(serial):
        ax.annotate(f"{txt:.2f}s", (i, serial[i]), textcoords="offset points", xytext=(0,10), ha='center', color='#C44E52', fontweight='bold')
    
    for i, txt in enumerate(parallel):
        ax.annotate(f"{txt:.2f}s", (i, parallel[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='#4C72B0', fontweight='bold')

    ax.set_title(f'Performance Comparison: {title_suffix}', fontsize=15, pad=20)
    ax.set_ylabel('Execution Time (Seconds)', fontsize=12)
    ax.set_xlabel('Filter Type', fontsize=12)
    ax.legend(loc='upper right')
    
    # Add grid for easier reading
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"line_{output_filename}.png", dpi=300)
    print(f"Generated line_{output_filename}.png")
    plt.close()

# ==========================================
# 2. RADAR CHART (Speedup) - You liked this
# ==========================================
def plot_radar(csv_file, title_suffix, output_filename):
    if not os.path.exists(csv_file):
        return

    df = pd.read_csv(csv_file)
    
    categories = df['Filter'].tolist()
    N = len(categories)
    
    # Close the loop for the radar chart
    values = df['Speedup'].tolist()
    values += values[:1]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # Draw axes
    plt.xticks(angles[:-1], categories, color='black', size=12, fontweight='bold')
    
    # Draw y-labels
    ax.set_rlabel_position(0)
    max_val = max(df['Speedup']) + 1
    plt.yticks(np.arange(1, max_val, 1), color="grey", size=8)
    plt.ylim(0, max_val)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#8EBA42')
    ax.fill(angles, values, '#8EBA42', alpha=0.25)
    
    plt.title(f'Speedup Factor (Higher is Better)\n{title_suffix}', size=16, color='black', y=1.1)
    
    # Add value labels
    for i, v in enumerate(df['Speedup']):
        angle_rad = angles[i]
        ax.text(angle_rad, v + 0.3, f"{v:.1f}x", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"radar_{output_filename}.png", dpi=300)
    print(f"Generated radar_{output_filename}.png")
    plt.close()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # 1. Image Graphs
    plot_line_filled('results_image.csv', 'Image Processing', 'image')
    plot_radar('results_image.csv', 'Image Processing', 'image')
    
    # 2. Video Graphs
    plot_line_filled('results_video_benchmark.csv', 'Video Processing', 'video')
    plot_radar('results_video_benchmark.csv', 'Video Processing', 'video')