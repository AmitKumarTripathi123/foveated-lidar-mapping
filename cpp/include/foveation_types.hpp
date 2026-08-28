#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace foveated_mapping {

struct ZoneConfig {
    float min_dist{0.0f};
    float max_dist{10.0f};
    float voxel_size{0.05f};
    std::string name{"Near-Field"};
};

struct FoveationConfig {
    float min_range{0.5f};
    float near_dist{10.0f};
    float near_voxel{0.05f};
    float mid_dist{40.0f};
    float mid_voxel{0.15f};
    float far_dist{100.0f};
    float far_voxel{0.50f};
};

struct ZoneStats {
    std::string zone_name;
    float min_dist;
    float max_dist;
    float voxel_size;
    int32_t input_count;
    int32_t output_count;
    float reduction_pct;
};

struct FoveationResult {
    std::vector<float> points; // Shape (M, 4) flattened
    std::vector<int64_t> labels; // Shape (M,) or empty
    int32_t original_count;
    int32_t foveated_count;
    float overall_reduction_pct;
    std::vector<ZoneStats> zone_stats;
    int32_t filtered_out_count;
};

} // namespace foveated_mapping
