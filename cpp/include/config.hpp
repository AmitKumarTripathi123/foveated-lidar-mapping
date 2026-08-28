#pragma once

#include <tuple>

namespace foveated_mapping {

struct GridConfig {
    float min_x{-50.0f};
    float max_x{50.0f};
    float min_y{-50.0f};
    float max_y{50.0f};
    float resolution{0.20f};
    int width{500};
    int height{500};

    GridConfig() {
        width = static_cast<int>((max_x - min_x) / resolution);
        height = static_cast<int>((max_y - min_y) / resolution);
    }

    GridConfig(float x0, float x1, float y0, float y1, float res)
        : min_x(x0), max_x(x1), min_y(y0), max_y(y1), resolution(res) {
        width = static_cast<int>((max_x - min_x) / resolution);
        height = static_cast<int>((max_y - min_y) / resolution);
    }
};

} // namespace foveated_mapping
