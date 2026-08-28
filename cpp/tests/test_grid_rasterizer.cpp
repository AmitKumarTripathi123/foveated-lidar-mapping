#include "grid_rasterizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace foveated_mapping;

int main() {
    GridConfig config(-10.0f, 10.0f, -10.0f, 10.0f, 1.0f);
    NativeGridRasterizer rasterizer(config);

    // Test points: (0.5, 0.5, 1.0), (0.5, 0.5, 2.0), (0.5, 0.5, 3.0) -> cell (10, 10)
    std::vector<float> xyz = {
        0.5f, 0.5f, 1.0f,
        0.5f, 0.5f, 2.0f,
        0.5f, 0.5f, 3.0f
    };
    std::vector<int64_t> classes = {0, 0, 1}; // Two Drivable, one Non-Drivable
    std::vector<float> conf = {0.8f, 0.9f, 0.7f};

    auto res = rasterizer.rasterize(xyz.data(), classes.data(), conf.data(), 3);

    // Cell at index (iy=10, ix=10) -> flat idx 10*20 + 10 = 210
    int idx = 10 * 20 + 10;
    assert(res.point_count_layer[idx] == 3);
    assert(std::abs(res.elevation_mean[idx] - 2.0f) < 1e-5f);
    assert(std::abs(res.elevation_min[idx] - 1.0f) < 1e-5f);
    assert(std::abs(res.elevation_max[idx] - 3.0f) < 1e-5f);
    assert(res.semantic_layer[idx] == 0); // Majority Drivable
    assert(res.traversability_layer[idx] == 1.0f);

    std::cout << "All C++ unit assertions PASSED!" << std::endl;
    return 0;
}
