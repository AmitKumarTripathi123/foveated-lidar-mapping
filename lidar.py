import numpy as np

bin_path = "dataset/sequences/00/velodyne/000000.bin"
label_path = "dataset/sequences/00/labels/000000.label"

points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
labels = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF

print("points:", points.shape)
print("labels:", labels.shape)
print("unique label ids:", np.unique(labels))