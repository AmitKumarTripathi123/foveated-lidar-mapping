#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>

namespace foveated_mapping {

enum SuperClass : uint8_t {
    DRIVABLE_TERRAIN = 0,
    NON_DRIVABLE_TERRAIN = 1,
    STATIC_OBSTACLE = 2,
    DYNAMIC_OBJECT = 3,
    IGNORE_LABEL = 255
};

struct ClassifiedPoint {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float intensity{0.0f};
    uint8_t class_id{SuperClass::IGNORE_LABEL};
    float confidence{1.0f};
};

struct FoveationBand {
    std::string name;
    float min_range{0.0f};
    float max_range{100.0f};
    float voxel_size{0.05f};

    bool contains(float r) const {
        return (r >= min_range && r < max_range);
    }
};

inline int get_semantic_priority(uint8_t class_id) {
    switch (class_id) {
        case SuperClass::DYNAMIC_OBJECT:       return 4;
        case SuperClass::STATIC_OBSTACLE:      return 3;
        case SuperClass::NON_DRIVABLE_TERRAIN: return 2;
        case SuperClass::DRIVABLE_TERRAIN:     return 1;
        default:                               return 0; // IGNORE_LABEL or undefined
    }
}

inline float calculate_traversability(uint8_t class_id) {
    if (class_id == SuperClass::DRIVABLE_TERRAIN) return 1.0f;
    if (class_id == SuperClass::NON_DRIVABLE_TERRAIN) return 0.2f;
    return 0.0f;
}

struct GridCell {
    std::string band_name;
    int64_t ix{0};
    int64_t iy{0};
    float resolution{0.05f};
    int64_t point_count{0};
    float elevation_mean{0.0f};
    float elevation_min{std::numeric_limits<float>::infinity()};
    float elevation_max{-std::numeric_limits<float>::infinity()};
    uint8_t semantic_class{SuperClass::IGNORE_LABEL};
    float confidence{0.0f};
    float traversability{0.0f};
    std::array<int64_t, 4> class_counts{0, 0, 0, 0};
    int64_t ignore_count{0};

    // Bounds helper
    float min_x() const { return static_cast<float>(ix) * resolution; }
    float max_x() const { return static_cast<float>(ix + 1) * resolution; }
    float min_y() const { return static_cast<float>(iy) * resolution; }
    float max_y() const { return static_cast<float>(iy + 1) * resolution; }
    float height_range() const { return (point_count > 0) ? (elevation_max - elevation_min) : 0.0f; }

    int64_t valid_semantic_count() const {
        return class_counts[0] + class_counts[1] + class_counts[2] + class_counts[3];
    }

    float class_probability(int class_id) const {
        int64_t valid = valid_semantic_count();
        if (valid <= 0) {
            if (class_id == SuperClass::IGNORE_LABEL && ignore_count > 0) return 1.0f;
            return 0.0f;
        }
        if (class_id < 0 || class_id >= 4) return 0.0f;
        return static_cast<float>(class_counts[class_id]) / static_cast<float>(valid);
    }

    int dominant_class() const {
        int64_t max_v = 0;
        int best_c = SuperClass::IGNORE_LABEL;
        int best_p = -1;
        for (int c = 0; c < 4; ++c) {
            if (class_counts[c] > max_v || (class_counts[c] == max_v && max_v > 0 && get_semantic_priority(c) > best_p)) {
                max_v = class_counts[c];
                best_c = c;
                best_p = get_semantic_priority(c);
            }
        }
        if (best_c != SuperClass::IGNORE_LABEL) {
            return best_c;
        }
        return SuperClass::IGNORE_LABEL;
    }
};



} // namespace foveated_mapping

