import pandas as pd
import matplotlib.pyplot as plt
import sys

def create_graphs():
    try:
        # Read the data from your C++ program's output
        df = pd.read_csv("results.csv")
    except FileNotFoundError:
        print("Error: results.csv not found.")
        print("Please run your C++ program (./multicore) first to generate it.")
        sys.exit(1)

    print("Data read successfully:")
    print(df)
    
    # Get the filter names for the x-axis
    filters = df['Filter']

    # --- Graph 1: Serial vs Parallel Execution Time (Line Plot) ---
    
    plt.figure(figsize=(10, 7))
    
    # Plot Serial time
    plt.plot(filters, df['SerialTime'], label='Serial Time', marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Plot Parallel time
    plt.plot(filters, df['ParallelTime'], label='Parallel Time', marker='s', linestyle='--', linewidth=2, markersize=8)
    
    # Add labels and title
    plt.title('Serial vs Parallel Execution Time by Filter', fontsize=16)
    plt.xlabel('Filter Type', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':') # Add a grid for easier reading
    
    # Add data point labels
    for i in range(len(filters)):
        plt.text(filters[i], df['SerialTime'][i], f" {df['SerialTime'][i]:.2f}s", va='bottom')
        plt.text(filters[i], df['ParallelTime'][i], f" {df['ParallelTime'][i]:.2f}s", va='top')
        
    plt.tight_layout()
    plt.savefig("execution_time_line_graph.png")
    print("\nGenerated execution_time_line_graph.png")

    # --- Graph 2: Speedup (Line Plot) ---
    
    plt.figure(figsize=(10, 7))
    
    # Plot Speedup
    plt.plot(filters, df['Speedup'], label='Speedup', marker='D', linestyle='-.', color='green', linewidth=2, markersize=8)

    # Add labels and title
    plt.title('Speedup Achieved by Parallelization', fontsize=16)
    plt.xlabel('Filter Type', fontsize=12)
    plt.ylabel('Speedup Factor (x)', fontsize=12)
    plt.grid(True, linestyle=':')
    
    # Add data point labels
    for i in range(len(filters)):
        plt.text(filters[i], df['Speedup'][i], f" {df['Speedup'][i]:.2f}x", va='bottom')

    plt.tight_layout()
    plt.savefig("speedup_line_graph.png")
    print("Generated speedup_line_graph.png")

if __name__ == "__main__":
    create_graphs()