#include "spvcnn_preprocessor.hpp"
#include <iostream>
#include <cassert>

using namespace foveated_mapping;

int main() {
    SPVCNNPreprocessor preprocessor(0.05f);

    // 3 points: P1 and P2 in same 5cm voxel (0.01, 0.01, 0.01) -> (0,0,0)
    // P3 in different voxel (0.10, 0.0, 0.0) -> (2,0,0)
    std::vector<float> points = {
        0.01f, 0.01f, 0.01f, 0.5f,
        0.02f, 0.02f, 0.02f, 0.6f,
        0.10f, 0.00f, 0.00f, 0.7f
    };

    auto res = preprocessor.quantize_and_index(points.data(), 3, 4);

    assert(res.num_points == 3);
    assert(res.num_voxels == 2);
    assert(res.point_to_voxel_idx[0] == 0);
    assert(res.point_to_voxel_idx[1] == 0);
    assert(res.point_to_voxel_idx[2] == 1);
    assert(res.voxel_to_point_idx[0] == 0);
    assert(res.voxel_to_point_idx[1] == 2);

    std::cout << "All C++ SPVCNN Preprocessor tests PASSED!" << std::endl;
    return 0;
}
