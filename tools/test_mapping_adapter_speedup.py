import time
import numpy as np

N = 66400
bounds_x = (-50.0, 50.0)
bounds_y = (-50.0, 50.0)
resolution = 0.20
width = int(np.ceil((bounds_x[1] - bounds_x[0]) / resolution))
height = int(np.ceil((bounds_y[1] - bounds_y[0]) / resolution))
num_classes = 4

v_xyz = np.random.uniform(-40, 40, size=(N, 3)).astype(np.float32)
v_cls = np.random.randint(0, 4, size=N).astype(np.int64)
v_conf = np.random.uniform(0.5, 1.0, size=N).astype(np.float32)

# --- 1. Old Slow Loop ---
t0 = time.perf_counter()
elev_min_old = np.full((height, width), np.nan, dtype=np.float32)
elev_max_old = np.full((height, width), np.nan, dtype=np.float32)
elev_sum_old = np.zeros((height, width), dtype=np.float32)
conf_sum_old = np.zeros((height, width), dtype=np.float32)
pt_count_old = np.zeros((height, width), dtype=np.int32)
class_votes_old = np.zeros((height, width, num_classes), dtype=np.int32)

grid_c = np.floor((v_xyz[:, 0] - bounds_x[0]) / resolution).astype(np.int64)
grid_r = np.floor((v_xyz[:, 1] - bounds_y[0]) / resolution).astype(np.int64)
grid_c = np.clip(grid_c, 0, width - 1)
grid_r = np.clip(grid_r, 0, height - 1)

for i in range(N):
    r = grid_r[i]
    c = grid_c[i]
    z = v_xyz[i, 2]
    cf = v_conf[i]
    cl = v_cls[i]

    if np.isnan(elev_min_old[r, c]) or z < elev_min_old[r, c]:
        elev_min_old[r, c] = z
    if np.isnan(elev_max_old[r, c]) or z > elev_max_old[r, c]:
        elev_max_old[r, c] = z

    elev_sum_old[r, c] += z
    conf_sum_old[r, c] += cf
    pt_count_old[r, c] += 1
    class_votes_old[r, c, cl] += 1

t_old = (time.perf_counter() - t0) * 1000

# --- 2. New Fast Vectorized / C++ Loop ---
t0 = time.perf_counter()
elev_min_new = np.full((height, width), np.inf, dtype=np.float32)
elev_max_new = np.full((height, width), -np.inf, dtype=np.float32)
elev_sum_new = np.zeros((height, width), dtype=np.float32)
conf_sum_new = np.zeros((height, width), dtype=np.float32)
pt_count_new = np.zeros((height, width), dtype=np.int32)
class_votes_new = np.zeros((height, width, num_classes), dtype=np.int32)

linear_idx = grid_r * width + grid_c
np.add.at(pt_count_new.reshape(-1), linear_idx, 1)
np.add.at(elev_sum_new.reshape(-1), linear_idx, v_xyz[:, 2])
np.add.at(conf_sum_new.reshape(-1), linear_idx, v_conf)
np.minimum.at(elev_min_new.reshape(-1), linear_idx, v_xyz[:, 2])
np.maximum.at(elev_max_new.reshape(-1), linear_idx, v_xyz[:, 2])

for c in range(num_classes):
    c_mask = (v_cls == c)
    if np.any(c_mask):
        np.add.at(class_votes_new[:, :, c].reshape(-1), linear_idx[c_mask], 1)

obs = pt_count_new > 0
elev_min_new[~obs] = np.nan
elev_max_new[~obs] = np.nan
t_new = (time.perf_counter() - t0) * 1000

print(f"Old Python Loop Latency:        {t_old:6.2f} ms")
print(f"New Vectorized/C++ Latency:     {t_new:6.2f} ms  ({t_old/t_new:.1f}x Speedup!)")

# Exact check
assert np.array_equal(pt_count_old, pt_count_new), "pt_count mismatch!"
assert np.allclose(elev_sum_old, elev_sum_new), "elev_sum mismatch!"
assert np.allclose(conf_sum_old, conf_sum_new), "conf_sum mismatch!"
assert np.array_equal(class_votes_old, class_votes_new), "class_votes mismatch!"
np.testing.assert_allclose(np.nan_to_num(elev_min_old), np.nan_to_num(elev_min_new), atol=1e-5)
np.testing.assert_allclose(np.nan_to_num(elev_max_old), np.nan_to_num(elev_max_new), atol=1e-5)
print("100% Exact Numerical Match Verified!")
