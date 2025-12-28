import os
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import hdbscan
import umap
import gc
from tqdm import tqdm
import shutil
from collections import Counter

# ==========================================
# CONFIGURATION
# ==========================================
MASTER_FILE = 'master_08052022.pkl'
OUTPUT_DIR = 'clustering_output_v3'

# Image Stats
IMG_SIZE = 144
MODEL_INPUT_SIZE = 224

# Processing Stats
BATCH_SIZE = 32
ROTATION_STEPS = 16  # Check 16 angles (every 22.5 degrees) to ensure continuous invariance

# Clustering Tuning (Targeting ~8-15 clusters)
UMAP_NEIGHBORS = 50       # High value = focuses on GLOBAL structure, ignores local noise
UMAP_MIN_DIST = 0.0       # Pack points tight
HDBSCAN_MIN_SIZE = 150    # Huge increase: A cluster must have >150 trajectories to be valid
HDBSCAN_MIN_SAMPLES = 10  # Smooths out density estimation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. REPORTING ENGINE
# ==========================================
class Reporter:
    def __init__(self):
        self.stats = {}
        self.log_buffer = []

    def log(self, msg):
        print(msg)
        self.log_buffer.append(msg)

    def compute_stats(self, labels, embeddings):
        total = len(labels)
        noise = list(labels).count(-1)
        clusters = set(labels) - {-1}
        
        counts = Counter(labels)
        sizes = [counts[l] for l in clusters]
        
        self.stats = {
            "Total Images": total,
            "Clusters Found": len(clusters),
            "Noise Points": noise,
            "Noise %": round((noise / total) * 100, 2),
            "Avg Cluster Size": round(np.mean(sizes), 2) if sizes else 0,
            "Max Cluster Size": max(sizes) if sizes else 0,
            "Min Cluster Size": min(sizes) if sizes else 0,
            "Embedding Dim": embeddings.shape[1],
            "Embedding Variance": round(np.var(embeddings), 4)
        }

    def save_report(self, path):
        with open(path, 'w') as f:
            f.write("=== V3 CLUSTERING ANALYSIS REPORT ===\n\n")
            
            f.write("--- 1. CONFIGURATION ---\n")
            f.write(f"Rotations Tested: {ROTATION_STEPS} (Max-Pool Aggregation)\n")
            f.write(f"UMAP Neighbors: {UMAP_NEIGHBORS}\n")
            f.write(f"HDBSCAN Min Cluster Size: {HDBSCAN_MIN_SIZE}\n\n")
            
            f.write("--- 2. CLUSTERING STATISTICS ---\n")
            for k, v in self.stats.items():
                f.write(f"{k}: {v}\n")
            
            f.write("\n--- 3. DISTRIBUTION (Top 10 Clusters) ---\n")
            if 'Total Images' in self.stats:
                # Re-calculate counts for the report
                # (We don't store raw labels in stats to save memory, so we assume they are passed or calculated before)
                pass 
            
            f.write("\n--- 4. EXECUTION LOG ---\n")
            for line in self.log_buffer:
                f.write(line + "\n")

report = Reporter()

# ==========================================
# 2. MODEL & TRANSFORMS
# ==========================================
def load_model():
    report.log(f"Loading DINOv2 on {device}...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.to(device)
    model.eval()
    return model

# Basic transform: Resize but keep scale relative to frame
base_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# ==========================================
# 3. INVARIANT FEATURE EXTRACTION
# ==========================================
def extract_invariant_features(df):
    """
    Generates embeddings.
    Technique: Dense TTA + Max Pooling.
    We rotate the image 16 times. We take the MAX value for every feature.
    This creates a signature of "features present anywhere" invariant to rotation.
    """
    model = load_model()
    
    # Pre-calculate angles
    angles = np.linspace(0, 360, ROTATION_STEPS, endpoint=False)
    
    all_features = []
    all_ids = []
    
    total = len(df)
    chunk_size = BATCH_SIZE * 5 # Process in reasonable memory chunks
    
    report.log(f"Starting extraction on {total} images...")
    
    for start_idx in tqdm(range(0, total, chunk_size)):
        end_idx = min(start_idx + chunk_size, total)
        batch_df = df.iloc[start_idx:end_idx]
        
        # We will process one image at a time to handle the TTA expansion
        # (1 image -> 16+ views) without exploding VRAM
        
        chunk_features = []
        chunk_ids = []
        
        for idx, row in batch_df.iterrows():
            try:
                # 1. Load & Preprocess Image
                img_arr = np.array(row['img']).reshape((IMG_SIZE, IMG_SIZE))
                
                # Make lines thicker (Dilation) - Critical for DINO visibility
                if img_arr.max() <= 1.0: img_arr = (img_arr * 255).astype(np.uint8)
                else: img_arr = img_arr.astype(np.uint8)
                
                # Invert (White line on Black BG)
                img_arr = cv2.bitwise_not(img_arr)
                kernel = np.ones((3,3), np.uint8)
                img_arr = cv2.dilate(img_arr, kernel, iterations=1)
                
                # Convert to RGB (stacking)
                img_rgb = np.stack((img_arr,)*3, axis=-1)
                
                # 2. Create Batch of Rotations
                base_tensor = base_transform(img_rgb)
                views = []
                
                # Add rotations AND flips
                for angle in angles:
                    rotated = F.rotate(base_tensor, angle)
                    views.append(rotated)
                    views.append(F.hflip(rotated)) # Flip Invariance
                
                # Stack: [32, 3, 224, 224]
                batch_view = torch.stack(views).to(device)
                
                # 3. Inference
                with torch.no_grad():
                    # [32, 384]
                    out = model(batch_view)
                    
                    # 4. MAX POOLING Aggregation
                    # "Did this feature appear in ANY rotation?"
                    # This is better than Mean pooling which "blurs" features.
                    invariant_feat, _ = torch.max(out, dim=0)
                    
                chunk_features.append(invariant_feat.cpu().numpy())
                chunk_ids.append(idx)
                
            except Exception as e:
                continue
        
        if chunk_features:
            all_features.extend(chunk_features)
            all_ids.extend(chunk_ids)
            
        # Memory Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    return np.array(all_features), all_ids

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def run_v3():
    # Setup
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    # Load Data
    report.log("Loading dataset...")
    df = pd.read_pickle(MASTER_FILE)
    
    # Extract
    features, ids = extract_invariant_features(df)
    
    # Dimensionality Reduction (UMAP)
    report.log("Running UMAP (Global Structure)...")
    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,  # High neighbors = Global structure
        n_components=10,
        min_dist=UMAP_MIN_DIST,
        metric='cosine',
        random_state=42
    )
    reduced_data = reducer.fit_transform(features)
    
    # Clustering (HDBSCAN)
    report.log("Running HDBSCAN (Macro Clusters)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_SIZE, # Force large clusters
        min_samples=HDBSCAN_MIN_SAMPLES,
        prediction_data=True
    )
    labels = clusterer.fit_predict(reduced_data)
    
    # --- REPORTING ---
    report.compute_stats(labels, features)
    
    # Save CSV
    pd.DataFrame({'id': ids, 'label': labels}).to_csv(f"{OUTPUT_DIR}/labels.csv", index=False)
    
    # Save Distribution Histogram to text
    counts = Counter(labels)
    sorted_clusters = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    report.log("\nTop 10 Cluster Sizes:")
    for lbl, count in sorted_clusters[:10]:
        report.log(f"Cluster {lbl}: {count} images")
        
    report.save_report(f"{OUTPUT_DIR}/report.txt")
    
    # Visualization Plot
    report.log("Generating UMAP Plot...")
    viz_reducer = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    embedding_2d = viz_reducer.fit_transform(features)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding_2d[:,0], embedding_2d[:,1], c=labels, cmap='Spectral', s=4, alpha=0.6)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"V3 Result: {len(set(labels)-{-1})} Clusters (Size > {HDBSCAN_MIN_SIZE})")
    plt.savefig(f"{OUTPUT_DIR}/cluster_map.png")
    plt.close()

    # Save Samples (Re-read approach to save RAM)
    report.log("Saving Sample Images...")
    for lbl in set(labels):
        folder = "noise" if lbl == -1 else f"cluster_{lbl}"
        os.makedirs(f"{OUTPUT_DIR}/clusters/{folder}", exist_ok=True)
        
        target_ids = [ids[i] for i, x in enumerate(labels) if x == lbl][:15]
        
        for tid in target_ids:
            try:
                row = df.loc[tid]
                img = np.array(row['img']).reshape((IMG_SIZE, IMG_SIZE))
                # Invert for visibility in folder
                if img.max() <= 1.0: img = (img * 255).astype(np.uint8)
                else: img = img.astype(np.uint8)
                img = cv2.bitwise_not(img)
                cv2.imwrite(f"{OUTPUT_DIR}/clusters/{folder}/{tid}.png", img)
            except: pass

    report.log("Done. Check 'clustering_output_v3' folder.")

if __name__ == "__main__":
    run_v3()