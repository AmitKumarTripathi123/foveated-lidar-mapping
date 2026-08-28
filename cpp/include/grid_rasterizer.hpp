#pragma once

#include "types.hpp"
#include "grid_accumulator.hpp"
#include "config.hpp"
#include <vector>

namespace foveated_mapping {

struct RasterizedGrid25D {
    int width;
    int height;
    std::vector<float> elevation_mean;
    std::vector<float> elevation_min;
    std::vector<float> elevation_max;
    std::vector<int64_t> semantic_layer;
    std::vector<float> confidence_layer;
    std::vector<float> traversability_layer;
    std::vector<int32_t> point_count_layer;
};

class NativeGridRasterizer {
public:
    explicit NativeGridRasterizer(const GridConfig& config = GridConfig());

    // Single-pass high-performance rasterization from contiguous pointer buffers
    RasterizedGrid25D rasterize(
        const float* xyz,           // shape (N, 3)
        const int64_t* classes,     // shape (N,)
        const float* confidences,   // shape (N,)
        size_t num_points
    ) const;

    const GridConfig& get_config() const { return config_; }

private:
    GridConfig config_;
};

} // namespace foveated_mapping
