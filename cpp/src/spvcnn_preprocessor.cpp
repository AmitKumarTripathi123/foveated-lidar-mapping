#include "spvcnn_preprocessor.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace foveated_mapping {

SPVCNNPreprocessor::SPVCNNPreprocessor(float voxel_size)
    : voxel_size_(voxel_size) {}

VoxelIndexResult SPVCNNPreprocessor::quantize_and_index(
    const float* points,
    size_t num_points,
    size_t stride
) const {
    VoxelIndexResult result;
    result.num_points = num_points;
    result.point_to_voxel_idx.resize(num_points);

    if (num_points == 0) {
        result.num_voxels = 0;
        return result;
    }

    size_t table_size = 1;
    while (table_size < num_points * 4) {
        table_size <<= 1;
    }
    if (table_size < 1024) table_size = 1024;
    const size_t table_mask = table_size - 1;

    std::vector<uint64_t> hash_keys(table_size, 0);
    std::vector<int32_t> hash_values(table_size, -1);
    std::vector<bool> hash_filled(table_size, false);

    result.voxel_coords.reserve(num_points * 3);
    result.voxel_to_point_idx.reserve(num_points);

    const float inv_v = 1.0f / voxel_size_;
    int32_t num_unique_voxels = 0;

    for (size_t i = 0; i < num_points; ++i) {
        float x = points[i * stride + 0];
        float y = points[i * stride + 1];
        float z = points[i * stride + 2];

        int64_t vx = static_cast<int64_t>(std::floor(x * inv_v));
        int64_t vy = static_cast<int64_t>(std::floor(y * inv_v));
        int64_t vz = static_cast<int64_t>(std::floor(z * inv_v));

        uint64_t ux = static_cast<uint64_t>(vx + 10000000) & 0x1FFFFF;
        uint64_t uy = static_cast<uint64_t>(vy + 10000000) & 0x1FFFFF;
        uint64_t uz = static_cast<uint64_t>(vz + 10000000) & 0x1FFFFF;

        uint64_t key = (ux << 42) | (uy << 21) | uz;
        if (key == 0) key = 1;

        uint64_t h = (key * 0x9E3779B97F4A7C15ULL) >> 32;
        size_t slot = static_cast<size_t>(h & table_mask);

        bool found = false;
        while (hash_filled[slot]) {
            if (hash_keys[slot] == key) {
                found = true;
                result.point_to_voxel_idx[i] = hash_values[slot];
                break;
            }
            slot = (slot + 1) & table_mask;
        }

        if (!found) {
            hash_filled[slot] = true;
            hash_keys[slot] = key;
            hash_values[slot] = num_unique_voxels;

            result.point_to_voxel_idx[i] = num_unique_voxels;
            result.voxel_to_point_idx.push_back(static_cast<int64_t>(i));
            result.voxel_coords.push_back(vx);
            result.voxel_coords.push_back(vy);
            result.voxel_coords.push_back(vz);

            num_unique_voxels++;
        }
    }

    result.num_voxels = static_cast<size_t>(num_unique_voxels);
    return result;
}

} // namespace foveated_mapping
