#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>

namespace foveated_mapping {

struct GridAccumulator {
    uint32_t count{0};
    float sum_z{0.0f};
    float min_z{std::numeric_limits<float>::infinity()};
    float max_z{-std::numeric_limits<float>::infinity()};
    float confidence_sum{0.0f};
    uint32_t class_counts[4]{0, 0, 0, 0};
    uint32_t ignore_count{0};

    inline void add_point(float z, uint8_t class_id, float conf) {
        count++;
        sum_z += z;
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
        confidence_sum += conf;

        if (class_id < 4) {
            class_counts[class_id]++;
        } else {
            ignore_count++;
        }
    }

    inline float get_elevation_mean() const {
        return count > 0 ? (sum_z / static_cast<float>(count)) : std::numeric_limits<float>::quiet_NaN();
    }

    inline float get_confidence_mean() const {
        return count > 0 ? (confidence_sum / static_cast<float>(count)) : 0.0f;
    }

    inline uint8_t get_dominant_semantic() const {
        if (count == 0) return 255;
        uint32_t best_cnt = 0;
        uint8_t best_c = 255;
        for (uint8_t c = 0; c < 4; ++c) {
            if (class_counts[c] > best_cnt) {
                best_cnt = class_counts[c];
                best_c = c;
            }
        }
        return best_c;
    }

    inline float get_traversability() const {
        uint8_t c = get_dominant_semantic();
        if (c == 0) return 1.0f;   // Drivable
        if (c == 1) return -1.0f;  // Non-Drivable
        if (c == 2 || c == 3) return 0.0f; // Obstacle / Dynamic
        return -1.0f; // Ignore / Unobserved
    }
};

} // namespace foveated_mapping
