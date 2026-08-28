#pragma once

#include "foveation_types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace foveated_mapping {

class FoveationAccelerator {
public:
    explicit FoveationAccelerator(const FoveationConfig& config = FoveationConfig());

    // Single-pass 3-zone distance calculation and open-addressing voxel deduplication
    FoveationResult foveate(
        const float* points,        // Shape (N, 4) flattened [x, y, z, intensity]
        const int64_t* labels,      // Shape (N,) or nullptr
        size_t num_points
    ) const;

    const FoveationConfig& get_config() const { return config_; }

private:
    FoveationConfig config_;

    // Helper to downsample a zone given pre-filtered point indices
    std::vector<int32_t> downsample_zone_indices(
        const float* points,
        const std::vector<int32_t>& zone_indices,
        float voxel_size
    ) const;
};

} // namespace foveated_mapping
