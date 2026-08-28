#include "grid_rasterizer.hpp"
#include <cmath>
#include <cstring>

namespace foveated_mapping {

NativeGridRasterizer::NativeGridRasterizer(const GridConfig& config)
    : config_(config) {}

RasterizedGrid25D NativeGridRasterizer::rasterize(
    const float* xyz,
    const int64_t* classes,
    const float* confidences,
    size_t num_points
) const {
    const int width = config_.width;
    const int height = config_.height;
    const size_t num_cells = static_cast<size_t>(width * height);
    const float inv_res = 1.0f / config_.resolution;

    std::vector<GridAccumulator> accumulators(num_cells);
    std::vector<uint32_t> active_indices;
    active_indices.reserve(std::min(num_points, num_cells));

    // Single-pass point traversal
    for (size_t i = 0; i < num_points; ++i) {
        const float x = xyz[i * 3 + 0];
        const float y = xyz[i * 3 + 1];
        const float z = xyz[i * 3 + 2];

        if (x < config_.min_x || x >= config_.max_x || y < config_.min_y || y >= config_.max_y) {
            continue;
        }

        const int ix = static_cast<int>(std::floor((x - config_.min_x) * inv_res));
        const int iy = static_cast<int>(std::floor((y - config_.min_y) * inv_res));

        if (ix < 0 || ix >= width || iy < 0 || iy >= height) {
            continue;
        }

        const size_t idx = static_cast<size_t>(iy * width + ix);
        auto& acc = accumulators[idx];

        if (acc.count == 0) {
            active_indices.push_back(static_cast<uint32_t>(idx));
        }

        const uint8_t c = (classes[i] >= 0 && classes[i] <= 3) ? static_cast<uint8_t>(classes[i]) : 255;
        const float conf = confidences[i];
        acc.add_point(z, c, conf);
    }

    // Allocate result layer arrays
    RasterizedGrid25D result;
    result.width = width;
    result.height = height;
    result.elevation_mean.assign(num_cells, std::numeric_limits<float>::quiet_NaN());
    result.elevation_min.assign(num_cells, std::numeric_limits<float>::quiet_NaN());
    result.elevation_max.assign(num_cells, std::numeric_limits<float>::quiet_NaN());
    result.semantic_layer.assign(num_cells, 255);
    result.confidence_layer.assign(num_cells, 0.0f);
    result.traversability_layer.assign(num_cells, -1.0f);
    result.point_count_layer.assign(num_cells, 0);

    // Finalize only non-empty active cells
    for (uint32_t idx : active_indices) {
        const auto& acc = accumulators[idx];
        if (acc.count > 0) {
            result.point_count_layer[idx] = static_cast<int32_t>(acc.count);
            result.elevation_mean[idx] = acc.get_elevation_mean();
            result.elevation_min[idx] = acc.min_z;
            result.elevation_max[idx] = acc.max_z;
            result.semantic_layer[idx] = static_cast<int64_t>(acc.get_dominant_semantic());
            result.confidence_layer[idx] = acc.get_confidence_mean();
            result.traversability_layer[idx] = acc.get_traversability();
        }
    }

    return result;
}

} // namespace foveated_mapping
