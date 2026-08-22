#include "foveated_grid.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace foveated_mapping;

void test_distance_and_bands() {
    FoveatedGridEngine engine;

    // Test band resolution
    const auto* b1 = engine.resolve_band(5.0f);
    assert(b1 != nullptr && b1->name == "near_field" && b1->voxel_size == 0.05f);

    const auto* b2 = engine.resolve_band(10.0f);
    assert(b2 != nullptr && b2->name == "mid_near_field" && b2->voxel_size == 0.10f);

    const auto* b3 = engine.resolve_band(30.0f);
    assert(b3 != nullptr && b3->name == "mid_far_field" && b3->voxel_size == 0.25f);

    const auto* b4 = engine.resolve_band(60.0f);
    assert(b4 != nullptr && b4->name == "far_field" && b4->voxel_size == 0.50f);

    // Out of bounds
    assert(engine.resolve_band(100.0f) == nullptr);
    assert(engine.resolve_band(120.0f) == nullptr);
    assert(engine.resolve_band(-1.0f) == nullptr);
    assert(engine.resolve_band(std::numeric_limits<float>::quiet_NaN()) == nullptr);

    std::cout << "[PASS] Distance & Band Resolution Selection Tests\n";
}

void test_cell_indexing() {
    // Near field (0.05m)
    auto [ix1, iy1] = FoveatedGridEngine::xy_to_cell(2.01f, 3.02f, 0.05f);
    assert(ix1 == 40 && iy1 == 60);

    // Negative coordinates (must use mathematical floor)
    auto [ix2, iy2] = FoveatedGridEngine::xy_to_cell(-0.01f, -0.04f, 0.05f);
    assert(ix2 == -1 && iy2 == -1);

    auto [ix3, iy3] = FoveatedGridEngine::xy_to_cell(-4.25f, -3.15f, 0.05f);
    assert(ix3 == -85 && iy3 == -63);

    std::cout << "[PASS] 2D Cell Indexing & Floor Math Tests\n";
}

void test_aggregation_and_priority() {
    FoveatedGridEngine engine;
    std::vector<ClassifiedPoint> pts;

    // 3 points in cell (40, 60): Drivable, Non-Drivable, Static Obstacle
    pts.push_back({2.01f, 3.02f, 1.0f, 0.5f, SuperClass::DRIVABLE_TERRAIN, 0.9f});
    pts.push_back({2.02f, 3.03f, 2.0f, 0.5f, SuperClass::NON_DRIVABLE_TERRAIN, 0.8f});
    pts.push_back({2.04f, 3.04f, 3.0f, 0.5f, SuperClass::STATIC_OBSTACLE, 0.95f});

    auto cells = engine.build_grid(pts);
    assert(cells.size() == 1);
    const auto& c = cells[0];
    assert(c.ix == 40 && c.iy == 60);
    assert(c.point_count == 3);
    assert(std::abs(c.elevation_mean - 2.0f) < 1e-5f);
    assert(std::abs(c.elevation_min - 1.0f) < 1e-5f);
    assert(std::abs(c.elevation_max - 3.0f) < 1e-5f);
    assert(c.semantic_class == SuperClass::STATIC_OBSTACLE); // Obstacle overrides terrain
    assert(c.traversability == 0.0f);

    std::cout << "[PASS] Cell Aggregation & Semantic Priority Hierarchy Tests\n";
}

void test_empty_and_out_of_bounds() {
    FoveatedGridEngine engine;
    std::vector<ClassifiedPoint> empty_pts;
    auto cells_empty = engine.build_grid(empty_pts);
    assert(cells_empty.empty());

    std::vector<ClassifiedPoint> invalid_pts = {
        {150.0f, 0.0f, 1.0f, 0.5f, SuperClass::DRIVABLE_TERRAIN, 1.0f},
        {std::numeric_limits<float>::quiet_NaN(), 2.0f, 1.0f, 0.5f, SuperClass::DRIVABLE_TERRAIN, 1.0f}
    };
    auto cells_invalid = engine.build_grid(invalid_pts);
    assert(cells_invalid.empty());

    std::cout << "[PASS] Empty & Invalid Input Safety Tests\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  RUNNING C++ FOVEATED GRID ENGINE UNIT TESTS\n";
    std::cout << "========================================================\n";
    test_distance_and_bands();
    test_cell_indexing();
    test_aggregation_and_priority();
    test_empty_and_out_of_bounds();
    std::cout << "========================================================\n";
    std::cout << "  ALL C++ UNIT TESTS PASSED (100% OK)\n";
    std::cout << "========================================================\n";
    return 0;
}
