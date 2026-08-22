#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <optional>
#include <cstddef>

namespace foveated_mapping {

struct InternalBand {
    std::string name;
    float min_range{0.0f};
    float max_range{100.0f};
    float min_range_sq{0.0f};
    float max_range_sq{10000.0f};
    float voxel_size{0.05f};
    float inv_voxel_size{20.0f};
    int16_t band_idx{0};

    bool contains(float r) const {
        return (r >= min_range && r < max_range);
    }

    bool contains_sq(float r2) const {
        return (r2 >= min_range_sq && r2 < max_range_sq);
    }
};

class FoveatedGridEngine {
public:
    FoveatedGridEngine();
    explicit FoveatedGridEngine(const std::vector<FoveationBand>& custom_bands);

    // Primary pipeline interface
    std::vector<GridCell> build_grid(const std::vector<ClassifiedPoint>& points) const;

    // High-Performance direct buffer interface (Zero intermediate allocations)
    std::vector<GridCell> build_grid_raw(
        const float* pts_ptr,
        size_t num_points,
        size_t feat_dim,
        const int64_t* lbls_ptr,
        const float* confs_ptr
    ) const;

    // Helper functions
    const std::vector<FoveationBand>& get_bands() const { return bands_; }
    const FoveationBand* resolve_band(float r) const;
    static std::pair<int64_t, int64_t> xy_to_cell(float x, float y, float resolution);

    // CSV I/O helpers for deterministic golden test evaluation
    static std::vector<ClassifiedPoint> load_points_csv(const std::string& filepath);
    static bool export_grid_csv(const std::vector<GridCell>& cells, const std::string& filepath);

private:
    void init_internal_bands();

    std::vector<FoveationBand> bands_;
    std::vector<InternalBand> fast_bands_;
};

} // namespace foveated_mapping
