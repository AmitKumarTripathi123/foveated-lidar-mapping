#include "foveation_accelerator.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace foveated_mapping;

int main() {
    FoveationConfig config;
    FoveationAccelerator accelerator(config);

    // Test points across zones:
    // P1: (2, 2, 0, 0.5) -> d2 = 8 < 100 (Near)
    // P2: (2, 2, 0, 0.6) -> same voxel (Near)
    // P3: (20, 0, 0, 0.7) -> d2 = 400 (Mid)
    // P4: (60, 0, 0, 0.8) -> d2 = 3600 (Far)
    // P5: (120, 0, 0, 0.9) -> d2 = 14400 > 10000 (Filtered)
    std::vector<float> points = {
        2.0f, 2.0f, 0.0f, 0.5f,
        2.0f, 2.0f, 0.0f, 0.6f,
        20.0f, 0.0f, 0.0f, 0.7f,
        60.0f, 0.0f, 0.0f, 0.8f,
        120.0f, 0.0f, 0.0f, 0.9f
    };

    auto res = accelerator.foveate(points.data(), nullptr, 5);

    assert(res.original_count == 5);
    assert(res.foveated_count == 3); // P1 retained, P2 deduplicated, P3 retained, P4 retained
    assert(res.filtered_out_count == 1); // P5 filtered
    assert(res.zone_stats[0].input_count == 2);
    assert(res.zone_stats[0].output_count == 1);
    assert(res.zone_stats[1].input_count == 1);
    assert(res.zone_stats[1].output_count == 1);
    assert(res.zone_stats[2].input_count == 1);
    assert(res.zone_stats[2].output_count == 1);

    std::cout << "All C++ Foveation Accelerator tests PASSED!" << std::endl;
    return 0;
}
