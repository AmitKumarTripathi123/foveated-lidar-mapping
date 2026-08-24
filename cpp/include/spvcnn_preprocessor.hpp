#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace foveated_mapping {

struct VoxelIndexResult {
    std::vector<int64_t> voxel_coords; // Shape (M, 3) flattened
    std::vector<int64_t> point_to_voxel_idx; // Shape (N,)
    std::vector<int64_t> voxel_to_point_idx; // Shape (M,)
    size_t num_points{0};
    size_t num_voxels{0};
};

class SPVCNNPreprocessor {
public:
    explicit SPVCNNPreprocessor(float voxel_size = 0.05f);

    // Single-pass open-addressing voxel quantization and coordinate mapping
    VoxelIndexResult quantize_and_index(
        const float* points, // Shape (N, 4) or (N, 3)
        size_t num_points,
        size_t stride = 4
    ) const;

    float get_voxel_size() const { return voxel_size_; }

private:
    float voxel_size_;
};

} // namespace foveated_mapping
