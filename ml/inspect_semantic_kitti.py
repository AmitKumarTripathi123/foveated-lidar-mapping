from pathlib import Path
import numpy as np

def load_lidar(bin_path: Path) -> np.ndarray:
    """Load a SemanticKITTI LiDAR scan."""
    if not bin_path.exists():
        raise FileNotFoundError(f"LiDAR file not found: {bin_path}")
    
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Invalid LiDAR file: {raw.size} float values is not divisible by 4.")
    
    points = raw.reshape(-1, 4)
    return points

def load_labels(label_path: Path) -> np.ndarray:
    """Load and decode SemanticKITTI semantic labels."""
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    
    raw_labels = np.fromfile(label_path, dtype=np.uint32)
    # Lower 16 bits contain the semantic label.
    semantic_labels = raw_labels & 0xFFFF
    return semantic_labels

def inspect_scan(bin_path: Path, label_path: Path) -> None:
    """Inspect one LiDAR scan and its semantic labels."""
    points = load_lidar(bin_path)
    labels = load_labels(label_path)
    
    print("\n========== SEMANTICKITTI INSPECTION ==========\n")
    print(f"LiDAR shape      : {points.shape}")
    print(f"Labels shape     : {labels.shape}")
    
    # Critical validation
    if len(points) != len(labels):
        raise ValueError(f"Point-label mismatch! Points={len(points)}, Labels={len(labels)}")
    
    print("\n[OK] Point-label alignment verified.")
    
    # XYZ
    xyz = points[:, :3]
    intensity = points[:, 3]
    
    print("\n---------- XYZ Statistics ----------")
    print(f"X range: {xyz[:, 0].min():.3f} → {xyz[:, 0].max():.3f}")
    print(f"Y range: {xyz[:, 1].min():.3f} → {xyz[:, 1].max():.3f}")
    print(f"Z range: {xyz[:, 2].min():.3f} → {xyz[:, 2].max():.3f}")
    
    print("\n---------- Intensity Statistics ----------")
    print(f"Min : {intensity.min():.3f}")
    print(f"Max : {intensity.max():.3f}")
    print(f"Mean: {intensity.mean():.3f}")
    
    # Label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n---------- Semantic Label Distribution ----------")
    for label_id, count in zip(unique_labels, counts):
        percentage = 100.0 * count / len(labels)
        print(f"Class {label_id:3d} : {count:8d} points ({percentage:6.2f}%)")
        
    print("\n===============================================\n")

if __name__ == "__main__":
    # CHANGE THESE PATHS
    BIN_PATH = Path("data/semantic_kitti/sequences/00/velodyne/000000.bin")
    LABEL_PATH = Path("data/semantic_kitti/sequences/00/labels/000000.label")
    
    inspect_scan(BIN_PATH, LABEL_PATH)