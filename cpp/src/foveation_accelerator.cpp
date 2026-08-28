#include "foveation_accelerator.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>

namespace foveated_mapping {

FoveationAccelerator::FoveationAccelerator(const FoveationConfig& config)
    : config_(config) {}

std::vector<int32_t> FoveationAccelerator::downsample_zone_indices(
    const float* points,
    const std::vector<int32_t>& zone_indices,
    float voxel_size
) const {
    const size_t n_pts = zone_indices.size();
    if (n_pts == 0) {
        return {};
    }

    // 1. Compute minimum bounding coordinate
    float min_x = points[zone_indices[0] * 4 + 0];
    float min_y = points[zone_indices[0] * 4 + 1];
    float min_z = points[zone_indices[0] * 4 + 2];

    for (size_t i = 1; i < n_pts; ++i) {
        int32_t p_idx = zone_indices[i];
        float px = points[p_idx * 4 + 0];
        float py = points[p_idx * 4 + 1];
        float pz = points[p_idx * 4 + 2];
        if (px < min_x) min_x = px;
        if (py < min_y) min_y = py;
        if (pz < min_z) min_z = pz;
    }

    // 2. Open-addressing hash table
    size_t table_size = 1;
    while (table_size < n_pts * 4) {
        table_size <<= 1;
    }
    if (table_size < 1024) table_size = 1024;
    const size_t table_mask = table_size - 1;

    std::vector<uint64_t> hash_keys(table_size, 0);
    std::vector<bool> hash_filled(table_size, false);

    std::vector<int32_t> retained_indices;
    retained_indices.reserve(n_pts);

    const float inv_voxel = 1.0f / voxel_size;

    for (size_t i = 0; i < n_pts; ++i) {
        int32_t p_idx = zone_indices[i];
        float px = points[p_idx * 4 + 0];
        float py = points[p_idx * 4 + 1];
        float pz = points[p_idx * 4 + 2];

        int64_t vx = static_cast<int64_t>(std::floor((px - min_x) * inv_voxel));
        int64_t vy = static_cast<int64_t>(std::floor((py - min_y) * inv_voxel));
        int64_t vz = static_cast<int64_t>(std::floor((pz - min_z) * inv_voxel));

        uint64_t ux = static_cast<uint64_t>(vx) & 0xFFFFF;
        uint64_t uy = static_cast<uint64_t>(vy) & 0xFFFFF;
        uint64_t uz = static_cast<uint64_t>(vz) & 0x7FFFF;

        uint64_t key = (ux << 38) | (uy << 19) | uz;
        if (key == 0) key = 1;

        uint64_t h = (key * 0x9E3779B97F4A7C15ULL) >> 32;
        size_t slot = static_cast<size_t>(h & table_mask);

        bool found = false;
        while (hash_filled[slot]) {
            if (hash_keys[slot] == key) {
                found = true;
                break;
            }
            slot = (slot + 1) & table_mask;
        }

        if (!found) {
            hash_filled[slot] = true;
            hash_keys[slot] = key;
            retained_indices.push_back(p_idx);
        }
    }

    return retained_indices;
}

FoveationResult FoveationAccelerator::foveate(
    const float* points,
    const int64_t* labels,
    size_t num_points
) const {
    FoveationResult result;
    result.original_count = static_cast<int32_t>(num_points);
    result.filtered_out_count = 0;

    if (num_points == 0) {
        result.foveated_count = 0;
        result.overall_reduction_pct = 0.0f;
        return result;
    }

    const float n_d2 = config_.near_dist * config_.near_dist;
    const float m_d2 = config_.mid_dist * config_.mid_dist;
    const float f_d2 = config_.far_dist * config_.far_dist;

    std::vector<int32_t> near_indices;
    std::vector<int32_t> mid_indices;
    std::vector<int32_t> far_indices;

    near_indices.reserve(num_points / 2);
    mid_indices.reserve(num_points / 2);
    far_indices.reserve(num_points / 4);

    // Single pass zone classification
    for (size_t i = 0; i < num_points; ++i) {
        float x = points[i * 4 + 0];
        float y = points[i * 4 + 1];
        float z = points[i * 4 + 2];
        float d2 = x * x + y * y + z * z;

        if (d2 >= 0.0f && d2 < n_d2) {
            near_indices.push_back(static_cast<int32_t>(i));
        } else if (d2 >= n_d2 && d2 < m_d2) {
            mid_indices.push_back(static_cast<int32_t>(i));
        } else if (d2 >= m_d2 && d2 <= f_d2) {
            far_indices.push_back(static_cast<int32_t>(i));
        } else {
            result.filtered_out_count++;
        }
    }

    // Downsample each zone
    auto near_retained = downsample_zone_indices(points, near_indices, config_.near_voxel);
    auto mid_retained = downsample_zone_indices(points, mid_indices, config_.mid_voxel);
    auto far_retained = downsample_zone_indices(points, far_indices, config_.far_voxel);

    size_t total_out = near_retained.size() + mid_retained.size() + far_retained.size();
    result.foveated_count = static_cast<int32_t>(total_out);
    result.points.resize(total_out * 4);
    if (labels != nullptr) {
        result.labels.resize(total_out);
    }

    // Build consolidated output arrays
    size_t out_idx = 0;
    auto copy_pts = [&](const std::vector<int32_t>& ret_list) {
        for (int32_t p_idx : ret_list) {
            result.points[out_idx * 4 + 0] = points[p_idx * 4 + 0];
            result.points[out_idx * 4 + 1] = points[p_idx * 4 + 1];
            result.points[out_idx * 4 + 2] = points[p_idx * 4 + 2];
            result.points[out_idx * 4 + 3] = points[p_idx * 4 + 3];
            if (labels != nullptr) {
                result.labels[out_idx] = labels[p_idx];
            }
            out_idx++;
        }
    };

    copy_pts(near_retained);
    copy_pts(mid_retained);
    copy_pts(far_retained);

    // Compute zone statistics
    auto get_reduc = [](int32_t in_c, int32_t out_c) -> float {
        return in_c > 0 ? ((static_cast<float>(in_c - out_c) / static_cast<float>(in_c)) * 100.0f) : 0.0f;
    };

    result.zone_stats = {
        {"Near-Field (0-10m)", 0.0f, config_.near_dist, config_.near_voxel, static_cast<int32_t>(near_indices.size()), static_cast<int32_t>(near_retained.size()), get_reduc(static_cast<int32_t>(near_indices.size()), static_cast<int32_t>(near_retained.size()))},
        {"Mid-Field (10-40m)", config_.near_dist, config_.mid_dist, config_.mid_voxel, static_cast<int32_t>(mid_indices.size()), static_cast<int32_t>(mid_retained.size()), get_reduc(static_cast<int32_t>(mid_indices.size()), static_cast<int32_t>(mid_retained.size()))},
        {"Far-Field (40-100m)", config_.mid_dist, config_.far_dist, config_.far_voxel, static_cast<int32_t>(far_indices.size()), static_cast<int32_t>(far_retained.size()), get_reduc(static_cast<int32_t>(far_indices.size()), static_cast<int32_t>(far_retained.size()))},
    };

    result.overall_reduction_pct = num_points > 0 ? ((static_cast<float>(num_points - total_out) / static_cast<float>(num_points)) * 100.0f) : 0.0f;

    return result;
}

} // namespace foveated_mapping
