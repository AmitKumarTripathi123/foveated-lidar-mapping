#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

namespace foveated_mapping {

class FoveatedGridEngine {
public:
    // Default constructor initialized with canonical 3-zone 5/15/50 cm tiers
    FoveatedGridEngine();
    explicit FoveatedGridEngine(const std::vector<FoveationBand>& custom_bands);

    // Primary pipeline interface
    std::vector<GridCell> build_grid(const std::vector<ClassifiedPoint>& points) const;

    // Helper functions
    const std::vector<FoveationBand>& get_bands() const { return bands_; }
    const FoveationBand* resolve_band(float r) const;
    static std::pair<int64_t, int64_t> xy_to_cell(float x, float y, float resolution);

    // CSV I/O helpers for deterministic golden test evaluation
    static std::vector<ClassifiedPoint> load_points_csv(const std::string& filepath);
    static bool export_grid_csv(const std::vector<GridCell>& cells, const std::string& filepath);

private:
    std::vector<FoveationBand> bands_;
};

} // namespace foveated_mapping
