# Invariant-ViT Density Clustering (IVDC)

This repository contains a novel unsupervised clustering pipeline designed to categorize trajectory images based on their geometric structure. The methodology leverages a self-supervised Vision Transformer (ViT) backbone, invariant feature engineering, and density-based clustering to discover groups of similar trajectories without requiring a pre-defined number of clusters.

This project was developed to overcome the limitations of prior CNN-based methods (such as SCAN) and traditional K-Means/KNN approaches, specifically addressing the need for rotation/flip invariance and automatic cluster discovery in line-based trajectory datasets.

## 1. Methodology

The core philosophy of this approach is **Invariant-ViT Density Clustering (IVDC)**. Instead of training a network to cluster (supervised/semi-supervised), we **extract** rich geometric features using a frozen foundation model, **force** invariance through mathematical aggregation, and **discover** clusters using manifold learning and density estimation.

### 1.1 Feature Backbone: DINOv2

We utilize **DINOv2** (specifically the `dinov2_vits14` variant), a Vision Transformer trained via self-supervision (distillation). Unlike CLIP or standard ResNets, DINOv2 is optimized to understand local geometry, layout, and shape structure rather than just semantic texture. This makes it particularly effective for analyzing binary trajectory shapes where geometry is the only signal.

### 1.2 Invariant Feature Engineering (Max-Pooling TTA)

Standard Vision Transformers are not inherently rotation-invariant. To ensure that a trajectory and its rotated version map to the same point in the embedding space, we implement a **Dense Test-Time Augmentation (TTA)** strategy with **Max-Pooling Aggregation**.

For an input image , we generate a set of views  by applying transformations:



where  represents a set of geometric transformations including rotations at  intervals (e.g., ) and horizontal flips.

We pass all views through the ViT encoder  to obtain a set of feature vectors. The final invariant embedding  is computed via element-wise maximization:

By using the **maximum** activation rather than the mean, we preserve the distinct "existence" of geometric features (e.g., a sharp "hook" or "loop") regardless of their orientation. This prevents the "feature blurring" effect common in mean-pooling strategies, where distinct shapes average out to generic vectors.

### 1.3 Dimensionality Reduction: UMAP

The raw ViT embeddings (dimension ) are sparse and high-dimensional, often suffering from the "curse of dimensionality" where distance metrics lose discriminative power. We project these embeddings onto a lower-dimensional manifold using **UMAP (Uniform Manifold Approximation and Projection)**.

UMAP constructs a high-dimensional graph representation of the data and optimizes a low-dimensional layout to preserve the topological structure.

* **Metric:** Cosine distance is used to measure similarity in the deep embedding space.
* **Neighbors ():** Set to a high value (e.g., 50) to prioritize preserving global structure over local noise, effectively merging micro-clusters into meaningful macro-groups.

### 1.4 Automatic Clustering: HDBSCAN

To identify clusters without specifying  (the number of clusters), we use **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**.

Unlike K-Means, HDBSCAN models clusters as high-density regions separated by low-density areas. It constructs a hierarchy of connected components and extracts stable clusters based on their persistence (lifetime) across density thresholds.

* **Noise Handling:** Points in low-density regions that do not belong to any stable cluster are explicitly labeled as Noise (). This effectively filters out outlier trajectories that would otherwise corrupt valid clusters.
* **Stability:** A cluster is only accepted if it persists over a significant range of density cuts, ensuring robustness against parameter tuning.

---

## 2. Repository Structure

```text
├── clustering_pipeline_v3.py   # Main pipeline: Feature extraction -> UMAP -> HDBSCAN
├── visualize_clusters_v3.py    # Visualization: Generates Grid and Pollock plots
├── master_08052022.pkl         # (Input) Master data file containing images & trajectories
├── clustering_output_v3/       # (Output) Generated artifacts from the pipeline
│   ├── labels.csv              # Mapped Cluster IDs for every image
│   ├── report.txt              # Detailed statistics of the run
│   ├── cluster_map.png         # 2D UMAP projection plot
│   └── clusters/               # Sample images organized by folder
└── visualization_results_v3/   # (Output) High-quality figures
    ├── grid_plot_images.png    # Grid of sample images per cluster
    └── pollock_subplots_raw.png # Raw trajectory plots (Pollock style)

```

## 3. Usage

### 3.1 Environment Setup

The code relies on PyTorch (with CUDA support), UMAP, HDBSCAN, and OpenCV.

```bash
# Create a fresh environment (Recommended)
conda create -n traj_clustering python=3.10
conda activate traj_clustering

# Install PyTorch with CUDA support (Ensure compatibility with your GPU driver)
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install pandas numpy matplotlib opencv-python umap-learn hdbscan tqdm

```

### 3.2 Running the Clustering Pipeline

To process the data, extract features, and generate clusters:

```bash
python clustering_pipeline_v3.py

```

* **Input:** Reads `master_08052022.pkl`.
* **Process:** Extracts rotation-invariant DINOv2 features, reduces dimensionality with UMAP, and clusters with HDBSCAN.
* **Output:** Saves results and a logic report to `clustering_output_v3/`.

### 3.3 Visualizing Results

To generate the Grid and Pollock plots:

```bash
python visualize_clusters_v3.py --n_grid 10 --n_pollock 200

```

* `--n_grid`: Number of images to display per row in the grid plot.
* `--n_pollock`: Number of raw trajectories to overlay in the density plots.

---

## 4. Results & Visualization

### 4.1 Grid Plot (Cluster Samples)

The **Grid Plot** displays randomly sampled images from each discovered cluster. This allows for a quick qualitative assessment of cluster coherence and shape consistency.

* **Rows:** Distinct Clusters (Cluster 0, Cluster 1, ...).
* **Columns:** Random samples belonging to that cluster.
* **Noise:** The final row (if present) displays outliers that did not fit any cluster.

<img width="1102" height="1687" alt="grid_plot_images" src="https://github.com/user-attachments/assets/3a85e816-6d71-4f28-bfe1-ac10e2cd831a" />

### 4.2 Pollock Plot (Trajectory Density)

The **Pollock Plot** visualizes the raw trajectory coordinates for each cluster. By overlaying hundreds of trajectories, we can observe the density and variance of shapes within a group. The color coding represents the physical quantity associated with the trajectory (e.g., Red=`0`, Blue=`1u`, Green=`100n`, Orange=`10n`).

* **Dynamic View:** Each subplot automatically scales to the min/max coordinates of its trajectories to ensure maximum visibility.
* **Y-Axis:** Inverted to match the image coordinate system (origin at top-left).

<img width="2239" height="1467" alt="pollock_subplots_raw" src="https://github.com/user-attachments/assets/171b3d3c-eea4-45f6-8166-7d74a588a231" />

### 4.3 Observations

* **Automatic Discovery:** The algorithm successfully identified distinct clusters (e.g., 11 clusters) without being forced to find a specific number, validating the density-based approach.
* **Rotation Sensitivity:** While the method uses TTA for invariance, the separation of diagonal lines from horizontal/vertical lines (e.g., Cluster 4 vs Cluster 9) suggests some residual sensitivity to orientation. This is likely due to aliasing artifacts on the pixel grid which make diagonal lines structurally distinct to the network even after rotation.
