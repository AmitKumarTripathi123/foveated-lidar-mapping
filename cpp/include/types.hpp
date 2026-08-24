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

struct CellKey {
    int level{0}; // 0: Near (5cm), 1: Mid (15cm), 2: Far (50cm)
    int64_t ix{0};
    int64_t iy{0};

    bool operator==(const CellKey& other) const {
        return level == other.level && ix == other.ix && iy == other.iy;
    }
};

struct FoveationBand {
    std::string name;
    float min_range{0.0f};
    float max_range{100.0f};
    float voxel_size{0.05f};
    int level{0};

    bool contains(float r) const {
        return (r >= min_range && r < max_range);
    }
};

struct GridCell {
    CellKey key;
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
    float traversability{-1.0f};

    float min_x() const { return static_cast<float>(ix) * resolution; }
    float max_x() const { return static_cast<float>(ix + 1) * resolution; }
    float min_y() const { return static_cast<float>(iy) * resolution; }
    float max_y() const { return static_cast<float>(iy + 1) * resolution; }
};

inline int get_semantic_priority(uint8_t class_id) {
    switch (class_id) {
        case DYNAMIC_OBJECT:      return 4;
        case STATIC_OBSTACLE:     return 3;
        case NON_DRIVABLE_TERRAIN: return 2;
        case DRIVABLE_TERRAIN:    return 1;
        default:                  return 0;
    }
}

inline float calculate_traversability(uint8_t class_id) {
    switch (class_id) {
        case DRIVABLE_TERRAIN:     return 1.0f;
        case NON_DRIVABLE_TERRAIN: return -1.0f;
        case STATIC_OBSTACLE:      return 0.0f;
        case DYNAMIC_OBJECT:       return 0.0f;
        default:                   return -1.0f;
    }
}

} // namespace foveated_mapping
