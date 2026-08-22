from typing import Tuple
"""
Synthetic LiDAR Scan Generator.
Generates realistic multi-frame LiDAR sequences with SemanticKITTI binary format (.bin / .label)
including road, sidewalks, obstacles, dynamic objects, out-of-range points, and edge cases.
"""

import os
from pathlib import Path
import numpy as np


def generate_synthetic_lidar_frame(
    num_beams: int = 64,
    points_per_beam: int = 1800,
    seed: int = 42,
    include_invalid: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a single realistic 3D LiDAR scan.
    Returns:
        points: np.ndarray float32 [N, 4] -> [x, y, z, intensity]
        labels: np.ndarray uint32 [N] -> raw SemanticKITTI class IDs
    """
    np.random.seed(seed)
    pts_list = []
    lbls_list = []

    # Sensor parameters
    sensor_height = 1.73  # meters above ground
    vertical_fov = (-24.8, 2.0)  # degrees
    elevations = np.linspace(np.radians(vertical_fov[0]), np.radians(vertical_fov[1]), num_beams)
    azimuths = np.linspace(-np.pi, np.pi, points_per_beam)

    # 1. Ground Plane (Road 40, Sidewalk 48, Terrain 72)
    for el in elevations:
        if el < 0:  # Beam pointing down towards ground
            r = -sensor_height / np.sin(el)
            # Filter ground points within realistic range (1m to 80m)
            valid_r = r[(r >= 1.0) & (r <= 80.0)]
            for dist in valid_r:
                # Sample azimuths for ground
                az = np.random.uniform(-np.pi, np.pi, size=min(points_per_beam // 4, 400))
                gx = dist * np.cos(el) * np.cos(az)
                gy = dist * np.cos(el) * np.sin(az)
                gz = -sensor_height + np.random.normal(0, 0.02, size=len(gx))
                gi = np.random.uniform(0.1, 0.4, size=len(gx))

                # Classify: road (|y| <= 4), sidewalk (4 < |y| <= 6), terrain (|y| > 6)
                glbl = np.zeros(len(gx), dtype=np.uint32)
                glbl[np.abs(gy) <= 4.0] = 40       # Road
                glbl[(np.abs(gy) > 4.0) & (np.abs(gy) <= 6.0)] = 48  # Sidewalk
                glbl[np.abs(gy) > 6.0] = 72       # Terrain

                # Slight curb height at sidewalk
                gz[(np.abs(gy) > 4.0) & (np.abs(gy) <= 6.0)] += 0.15

                pts_list.append(np.column_stack([gx, gy, gz, gi]))
                lbls_list.append(glbl)

    # 2. Buildings and Walls (Class 50) at y = +-12m to +-25m
    for side in [-1, 1]:
        bx = np.random.uniform(-30, 80, size=8000)
        by = side * np.random.uniform(12, 25, size=8000)
        bz = np.random.uniform(-sensor_height, 6.0, size=8000)
        bi = np.random.uniform(0.2, 0.8, size=8000)
        blbl = np.full(8000, 50, dtype=np.uint32)  # Building
        pts_list.append(np.column_stack([bx, by, bz, bi]))
        lbls_list.append(blbl)

    # 3. Static Obstacles: Poles (80), Traffic Signs (81), Vegetation (70)
    # Poles along sidewalk at x = [5, 15, 25, 45, 65], y = +-5.5
    for px in [5.0, 15.0, 25.0, 45.0, 65.0]:
        for py in [-5.5, 5.5]:
            pz = np.linspace(-sensor_height, 2.5, 60)
            px_pts = px + np.random.normal(0, 0.04, size=len(pz))
            py_pts = py + np.random.normal(0, 0.04, size=len(pz))
            pi_pts = np.random.uniform(0.6, 0.9, size=len(pz))
            plbl = np.full(len(pz), 80, dtype=np.uint32)  # Pole
            pts_list.append(np.column_stack([px_pts, py_pts, pz, pi_pts]))
            lbls_list.append(plbl)

    # Vegetation (Trees) around x = [10, 30, 50, 70], y = +-8
    for tx in [10.0, 30.0, 50.0, 70.0]:
        for ty in [-8.0, 8.0]:
            tree_pts = np.random.normal([tx, ty, 1.5], [1.2, 1.2, 1.0], size=(400, 3))
            tree_pts[:, 2] = np.clip(tree_pts[:, 2], -sensor_height, 5.0)
            ti = np.random.uniform(0.1, 0.5, size=len(tree_pts))
            tlbl = np.full(len(tree_pts), 70, dtype=np.uint32)  # Vegetation
            pts_list.append(np.column_stack([tree_pts, ti]))
            lbls_list.append(tlbl)

    # 4. Dynamic Objects:
    # Car 1 (Near): x = 8.0, y = 1.8 (Car class 10)
    # Car 2 (Mid): x = 28.0, y = -1.8 (Car class 10)
    # Car 3 (Far): x = 68.0, y = 2.0 (Car class 10)
    # Pedestrian 1 (Near): x = 6.0, y = -4.5 (Person class 30)
    # Pedestrian 2 (Mid): x = 22.0, y = 4.8 (Person class 30)
    # Pedestrian 3 (Far): x = 55.0, y = -5.0 (Person class 30)
    # Bicyclist (Mid): x = 16.0, y = -3.2 (Bicyclist class 31)

    objects = [
        ("car", 8.0, 1.8, 10, [2.0, 1.0, 0.7], 500),
        ("car", 28.0, -1.8, 10, [2.0, 1.0, 0.7], 300),
        ("car", 68.0, 2.0, 10, [2.0, 1.0, 0.7], 80),
        ("pedestrian", 6.0, -4.5, 30, [0.3, 0.3, 0.8], 150),
        ("pedestrian", 22.0, 4.8, 30, [0.3, 0.3, 0.8], 60),
        ("pedestrian", 55.0, -5.0, 30, [0.3, 0.3, 0.8], 25),
        ("bicyclist", 16.0, -3.2, 31, [0.8, 0.4, 0.7], 120),
    ]

    for obj_type, ox, oy, raw_cls, dims, count in objects:
        dx, dy, dz = dims
        obj_xyz = np.random.uniform(
            [ox - dx, oy - dy, -sensor_height + 0.1],
            [ox + dx, oy + dy, -sensor_height + 0.1 + 2 * dz],
            size=(count, 3)
        )
        oi = np.random.uniform(0.4, 0.95, size=count)
        olbl = np.full(count, raw_cls, dtype=np.uint32)
        pts_list.append(np.column_stack([obj_xyz, oi]))
        lbls_list.append(olbl)

    # 5. Distant points (> 100m) to test range filtering (e.g. at 105m to 125m)
    far_x = np.random.uniform(102, 120, size=500)
    far_y = np.random.uniform(-40, 40, size=500)
    far_z = np.random.uniform(-sensor_height, 10.0, size=500)
    far_i = np.random.uniform(0.1, 0.5, size=500)
    far_lbl = np.full(500, 50, dtype=np.uint32)  # Distant buildings
    pts_list.append(np.column_stack([far_x, far_y, far_z, far_i]))
    lbls_list.append(far_lbl)

    all_pts = np.vstack(pts_list).astype(np.float32)
    all_lbls = np.concatenate(lbls_list).astype(np.uint32)

    # 6. Inject invalid coordinates if requested (to test validation)
    if include_invalid:
        # Inject 15 NaNs and 5 Infs
        nan_indices = np.random.choice(len(all_pts), 15, replace=False)
        all_pts[nan_indices, 0] = np.nan
        inf_indices = np.random.choice(len(all_pts), 5, replace=False)
        all_pts[inf_indices, 1] = np.inf

    return all_pts, all_lbls


def create_dataset_sequence(
    output_dir: Path,
    sequence_id: str = "00",
    num_frames: int = 5
):
    """Creates a SemanticKITTI compliant sequence directory."""
    seq_dir = output_dir / "sequences" / sequence_id
    velodyne_dir = seq_dir / "velodyne"
    labels_dir = seq_dir / "labels"

    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_frames} LiDAR frames in {seq_dir}...")

    for i in range(num_frames):
        frame_id = f"{i:06d}"
        include_inv = (i == 4)  # Frame 4 has edge case invalid points for testing
        pts, lbls = generate_synthetic_lidar_frame(seed=42 + i, include_invalid=include_inv)

        bin_file = velodyne_dir / f"{frame_id}.bin"
        label_file = labels_dir / f"{frame_id}.label"

        # Save binary float32
        pts.tofile(str(bin_file))

        # Save binary uint32 (semantic label in lower 16 bits)
        raw_label_data = lbls.astype(np.uint32)
        raw_label_data.tofile(str(label_file))

        print(f"  -> Frame {frame_id}: {len(pts):,} points, saved to {bin_file.name}")

    # Also create a corrupted frame in sequence 99 to test STRICT_STOP / mismatch handling
    corrupt_seq = output_dir / "sequences" / "99"
    c_velo = corrupt_seq / "velodyne"
    c_lbl = corrupt_seq / "labels"
    c_velo.mkdir(parents=True, exist_ok=True)
    c_lbl.mkdir(parents=True, exist_ok=True)

    pts, lbls = generate_synthetic_lidar_frame(seed=999)
    pts.tofile(str(c_velo / "000000.bin"))
    # Save mismatched labels length (fewer labels than points)
    mismatched_lbls = lbls[: len(lbls) - 500]
    mismatched_lbls.tofile(str(c_lbl / "000000.label"))
    print(f"Created test mismatch frame in sequence 99.")


if __name__ == "__main__":
    from typing import Tuple
    out_base = Path("data/synthetic_sequence")
    create_dataset_sequence(out_base, sequence_id="00", num_frames=5)
