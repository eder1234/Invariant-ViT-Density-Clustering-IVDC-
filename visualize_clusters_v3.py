import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import math

# ==========================================
# CONFIGURATION
# ==========================================
COLOR_MAP = {
    '0':    'red',
    '1u':   'blue',
    '100n': 'green',
    '10n':  'orange'
}
DEFAULT_COLOR = 'gray'
IMG_SIZE = 144

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize clustering results.")
    parser.add_argument('--master_pkl', type=str, default='master_08052022.pkl', help='Path to master data')
    parser.add_argument('--labels_csv', type=str, default='clustering_output_v3/labels.csv', help='Path to labels CSV')
    parser.add_argument('--output_dir', type=str, default='visualization_results_v3', help='Folder to save plots')
    parser.add_argument('--n_grid', type=int, default=10, help='Samples per cluster for Grid Plot')
    parser.add_argument('--n_pollock', type=int, default=200, help='Samples per cluster for Pollock Plot')
    return parser.parse_args()

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_data(master_path, labels_path):
    print(f"Loading data...")
    if not os.path.exists(master_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Input files not found.")
    df_master = pd.read_pickle(master_path)
    df_labels = pd.read_csv(labels_path)
    # Merge on index/id
    df = df_master.merge(df_labels, left_index=True, right_on='id')
    return df

# ==========================================
# 2. GRID PLOT (Images)
# ==========================================
def generate_grid_plot(df, n_samples, output_dir):
    print("Generating Grid Plot (Images)...")
    unique_labels = sorted(df['label'].unique())
    if -1 in unique_labels:
        unique_labels.remove(-1)
        unique_labels.append(-1)
        
    n_clusters = len(unique_labels)
    
    # Dynamic figure size
    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(n_samples, n_clusters * 1.2))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    
    # Handle 1D axes cases
    if n_clusters == 1: axes = np.array([axes])
    if n_samples == 1: axes = axes.reshape(n_clusters, 1)
    
    for row_idx, label in enumerate(tqdm(unique_labels, desc="Grid Rows")):
        cluster_data = df[df['label'] == label]
        
        # Sampling
        if len(cluster_data) >= n_samples:
            samples = cluster_data.sample(n=n_samples, random_state=42)
        else:
            samples = cluster_data.sample(n=n_samples, replace=True, random_state=42)
            
        col_idx = 0
        for _, row in samples.iterrows():
            ax = axes[row_idx, col_idx]
            img = np.array(row['img']).reshape((IMG_SIZE, IMG_SIZE))
            if img.max() <= 1.0: img = img * 255
            
            # Plot White BG, Black Lines
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            
            # Row Label
            if col_idx == 0:
                name = "Noise" if label == -1 else f"Cluster {label}"
                ax.text(-0.2, 0.5, name, transform=ax.transAxes, 
                        va='center', ha='right', fontsize=10, fontweight='bold')
            col_idx += 1
            
    out_path = os.path.join(output_dir, 'grid_plot_images.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

# ==========================================
# 3. POLLOCK PLOTS (Raw Trajectories - Subplots)
# ==========================================
def generate_pollock_subplots(df, n_samples, output_dir):
    print(f"Generating Pollock Subplots (n={n_samples})...")
    
    unique_labels = sorted(df['label'].unique())
    # Move noise to end
    if -1 in unique_labels:
        unique_labels.remove(-1)
        unique_labels.append(-1)
        
    n_total = len(unique_labels)
    
    # Calculate grid dimensions (approx square)
    n_cols = math.ceil(math.sqrt(n_total))
    n_rows = math.ceil(n_total / n_cols)
    
    # Create large figure for subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten() if n_total > 1 else [axes]
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for i, ax in enumerate(tqdm(axes, desc="Pollock Subplots")):
        # Hide unused subplots if n_total is not a perfect square
        if i >= n_total:
            ax.axis('off')
            continue
            
        label = unique_labels[i]
        cluster_data = df[df['label'] == label]
        
        # Sample data
        current_n = min(len(cluster_data), n_samples)
        samples = cluster_data.sample(n=current_n, random_state=42)
        
        # Initialize dynamic limits for this cluster
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        # Plot Loop
        for _, row in samples.iterrows():
            qty = str(row['quantity'])
            color = COLOR_MAP.get(qty, DEFAULT_COLOR)
            
            # Get Raw Trajectory [x_array, y_array]
            traj = row['traj']
            x, y = traj[0], traj[1]
            
            # Update dynamic limits
            min_x = min(min_x, x.min())
            max_x = max(max_x, x.max())
            min_y = min(min_y, y.min())
            max_y = max(max_y, y.max())
            
            # Plot line
            ax.plot(x, y, color=color, alpha=0.5, linewidth=1)

        # Set Dynamic Limits (with a little padding)
        pad_x = (max_x - min_x) * 0.05 if max_x != min_x else 5
        pad_y = (max_y - min_y) * 0.05 if max_y != min_y else 5
        
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        # Invert Y-axis: set ylim from MAX to MIN
        ax.set_ylim(max_y + pad_y, min_y - pad_y)
        
        ax.axis('off') # Clean look, just shapes
        
        title = "Noise" if label == -1 else f"Cluster {label}"
        ax.set_title(f"{title}\n(n={current_n})", fontsize=10)

    # Add Global Legend outside the subplots
    handles = [mlines.Line2D([], [], color=c, lw=2, label=q) for q, c in COLOR_MAP.items()]
    fig.legend(handles=handles, title="Quantity", loc='center right', bbox_to_anchor=(1.05, 0.5))
    
    # Save single giant figure
    out_path = os.path.join(output_dir, 'pollock_subplots_raw.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved Pollock Overview: {out_path}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        df = load_data(args.master_pkl, args.labels_csv)
        generate_grid_plot(df, args.n_grid, args.output_dir)
        generate_pollock_subplots(df, args.n_pollock, args.output_dir)
        print("\nVisualization complete!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()