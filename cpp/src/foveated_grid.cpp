#include "foveated_grid.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace foveated_mapping {

namespace {

struct CellAccumulator {
    CellKey key;
    std::string band_name;
    int64_t ix{0};
    int64_t iy{0};
    float resolution{0.05f};
    int64_t count{0};
    double sum_z{0.0};
    float min_z{std::numeric_limits<float>::infinity()};
    float max_z{-std::numeric_limits<float>::infinity()};
    double sum_conf{0.0};
    uint8_t best_class{SuperClass::IGNORE_LABEL};
    int best_priority{-1};

    void add_point(float z, uint8_t class_id, float conf) {
        count++;
        sum_z += z;
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);
        sum_conf += conf;

        int priority = get_semantic_priority(class_id);
        if (priority > best_priority) {
            best_priority = priority;
            best_class = class_id;
        }
    }

    GridCell to_grid_cell() const {
        GridCell cell;
        cell.key = key;
        cell.band_name = band_name;
        cell.ix = ix;
        cell.iy = iy;
        cell.resolution = resolution;
        cell.point_count = count;
        cell.elevation_mean = (count > 0) ? static_cast<float>(sum_z / count) : 0.0f;
        cell.elevation_min = min_z;
        cell.elevation_max = max_z;
        cell.semantic_class = best_class;
        cell.confidence = (count > 0) ? static_cast<float>(sum_conf / count) : 0.0f;
        cell.traversability = calculate_traversability(best_class);
        return cell;
    }
};

} // anonymous namespace

FoveatedGridEngine::FoveatedGridEngine() {
    // Canonical 3-Zone Distance Tiers matching configs/system_config.yaml
    bands_ = {
        {"near_zone", 0.0f,  10.0f, 0.05f, 0},
        {"mid_zone",  10.0f, 40.0f, 0.15f, 1},
        {"far_zone",  40.0f, 100.0f, 0.50f, 2}
    };
}

FoveatedGridEngine::FoveatedGridEngine(const std::vector<FoveationBand>& custom_bands)
    : bands_(custom_bands) {}

const FoveationBand* FoveatedGridEngine::resolve_band(float r) const {
    if (!std::isfinite(r) || r < 0.0f || r > 100.0f) {
        return nullptr;
    }
    for (const auto& band : bands_) {
        if (band.contains(r)) {
            return &band;
        }
    }
    if (std::abs(r - 100.0f) < 1e-3f && !bands_.empty()) {
        return &bands_.back();
    }
    return nullptr;
}

std::pair<int64_t, int64_t> FoveatedGridEngine::xy_to_cell(float x, float y, float resolution) {
    int64_t ix = static_cast<int64_t>(std::floor(x / resolution));
    int64_t iy = static_cast<int64_t>(std::floor(y / resolution));
    return {ix, iy};
}

std::vector<GridCell> FoveatedGridEngine::build_grid(const std::vector<ClassifiedPoint>& points) const {
    std::unordered_map<std::string, CellAccumulator> grid_map;

    for (const auto& pt : points) {
        float r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        const FoveationBand* band = resolve_band(r);
        if (!band) {
            continue;
        }

        auto [ix, iy] = xy_to_cell(pt.x, pt.y, band->voxel_size);
        std::string key_str = std::to_string(band->level) + "_" + std::to_string(ix) + "_" + std::to_string(iy);

        auto it = grid_map.find(key_str);
        if (it == grid_map.end()) {
            CellAccumulator acc;
            acc.key = CellKey{band->level, ix, iy};
            acc.band_name = band->name;
            acc.ix = ix;
            acc.iy = iy;
            acc.resolution = band->voxel_size;
            acc.add_point(pt.z, pt.class_id, pt.confidence);
            grid_map.emplace(key_str, acc);
        } else {
            it->second.add_point(pt.z, pt.class_id, pt.confidence);
        }
    }

    std::vector<GridCell> result;
    result.reserve(grid_map.size());
    for (const auto& [k, acc] : grid_map) {
        result.push_back(acc.to_grid_cell());
    }

    return result;
}

std::vector<ClassifiedPoint> FoveatedGridEngine::load_points_csv(const std::string& filepath) {
    std::vector<ClassifiedPoint> points;
    std::ifstream file(filepath);
    if (!file.is_open()) return points;

    std::string line;
    std::getline(file, line); // Skip header if present

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        ClassifiedPoint pt;

        std::getline(ss, val, ','); pt.x = std::stof(val);
        std::getline(ss, val, ','); pt.y = std::stof(val);
        std::getline(ss, val, ','); pt.z = std::stof(val);
        std::getline(ss, val, ','); pt.intensity = std::stof(val);
        std::getline(ss, val, ','); pt.class_id = static_cast<uint8_t>(std::stoi(val));
        std::getline(ss, val, ','); pt.confidence = std::stof(val);

        points.push_back(pt);
    }
    return points;
}

bool FoveatedGridEngine::export_grid_csv(const std::vector<GridCell>& cells, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) return false;

    file << "level,band_name,ix,iy,resolution,point_count,elevation_mean,elevation_min,elevation_max,semantic_class,confidence,traversability\n";
    for (const auto& c : cells) {
        file << c.key.level << ","
             << c.band_name << ","
             << c.ix << ","
             << c.iy << ","
             << c.resolution << ","
             << c.point_count << ","
             << c.elevation_mean << ","
             << c.elevation_min << ","
             << c.elevation_max << ","
             << static_cast<int>(c.semantic_class) << ","
             << c.confidence << ","
             << c.traversability << "\n";
    }
    return true;
}

} // namespace foveated_mapping
