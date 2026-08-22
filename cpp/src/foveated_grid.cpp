#include "foveated_grid.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace foveated_mapping {

namespace {

// Internal accumulator struct for cell point aggregation
struct CellAccumulator {
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
    bands_ = {
        {"near_field",     0.0f,  10.0f, 0.05f},
        {"mid_near_field", 10.0f, 30.0f, 0.10f},
        {"mid_far_field",  30.0f, 60.0f, 0.25f},
        {"far_field",      60.0f, 100.0f, 0.50f}
    };
}

FoveatedGridEngine::FoveatedGridEngine(const std::vector<FoveationBand>& custom_bands)
    : bands_(custom_bands) {}

const FoveationBand* FoveatedGridEngine::resolve_band(float r) const {
    if (!std::isfinite(r) || r < 0.0f || r >= 100.0f) {
        return nullptr;
    }
    for (const auto& band : bands_) {
        if (band.contains(r)) {
            return &band;
        }
    }
    return nullptr;
}

std::pair<int64_t, int64_t> FoveatedGridEngine::xy_to_cell(float x, float y, float resolution) {
    int64_t ix = static_cast<int64_t>(std::floor(x / resolution));
    int64_t iy = static_cast<int64_t>(std::floor(y / resolution));
    return {ix, iy};
}

std::vector<GridCell> FoveatedGridEngine::build_grid(const std::vector<ClassifiedPoint>& points) const {
    // Hash key: (band_index << 48) ^ (ix_shifted << 24) ^ (iy_shifted)
    // Or string key: band_name:ix:iy
    std::unordered_map<std::string, CellAccumulator> accumulators;
    accumulators.reserve(points.size());

    for (const auto& pt : points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            continue; // Skip non-finite points
        }

        float r = std::sqrt(pt.x * pt.x + pt.y * pt.y);
        const FoveationBand* band = resolve_band(r);
        if (band == nullptr) {
            continue; // Out of range [0, 100m)
        }

        auto [ix, iy] = xy_to_cell(pt.x, pt.y, band->voxel_size);

        std::string key = band->name + ":" + std::to_string(ix) + ":" + std::to_string(iy);
        auto it = accumulators.find(key);
        if (it == accumulators.end()) {
            CellAccumulator acc;
            acc.band_name = band->name;
            acc.ix = ix;
            acc.iy = iy;
            acc.resolution = band->voxel_size;
            acc.add_point(pt.z, pt.class_id, pt.confidence);
            accumulators.emplace(key, acc);
        } else {
            it->second.add_point(pt.z, pt.class_id, pt.confidence);
        }
    }

    std::vector<GridCell> result;
    result.reserve(accumulators.size());
    for (const auto& [_, acc] : accumulators) {
        result.push_back(acc.to_grid_cell());
    }

    // Deterministic sorting matching Python reference: (band_name, iy, ix)
    std::sort(result.begin(), result.end(), [](const GridCell& a, const GridCell& b) {
        if (a.band_name != b.band_name) return a.band_name < b.band_name;
        if (a.iy != b.iy) return a.iy < b.iy;
        return a.ix < b.ix;
    });

    return result;
}

std::vector<ClassifiedPoint> FoveatedGridEngine::load_points_csv(const std::string& filepath) {
    std::vector<ClassifiedPoint> points;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open input points CSV: " << filepath << "\n";
        return points;
    }

    auto parse_float = [](const std::string& s) -> float {
        if (s.empty() || s == "nan" || s == "NaN" || s == "NAN") {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (s == "inf" || s == "Inf" || s == "INFINITY" || s == "infinity") {
            return std::numeric_limits<float>::infinity();
        }
        if (s == "-inf" || s == "-Inf" || s == "-INFINITY" || s == "-infinity") {
            return -std::numeric_limits<float>::infinity();
        }
        try {
            return std::stof(s);
        } catch (...) {
            return std::numeric_limits<float>::quiet_NaN();
        }
    };

    std::string line;
    // Read header
    if (!std::getline(file, line)) return points;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        ClassifiedPoint pt;

        // x, y, z, intensity, class_id, confidence
        if (std::getline(ss, val, ',')) pt.x = parse_float(val);
        if (std::getline(ss, val, ',')) pt.y = parse_float(val);
        if (std::getline(ss, val, ',')) pt.z = parse_float(val);
        if (std::getline(ss, val, ',')) pt.intensity = parse_float(val);
        if (std::getline(ss, val, ',')) {
            try {
                pt.class_id = val.empty() ? SuperClass::IGNORE_LABEL : static_cast<uint8_t>(std::stoi(val));
            } catch (...) {
                pt.class_id = SuperClass::IGNORE_LABEL;
            }
        }
        if (std::getline(ss, val, ',')) pt.confidence = parse_float(val);

        points.push_back(pt);
    }
    return points;
}

bool FoveatedGridEngine::export_grid_csv(const std::vector<GridCell>& cells, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open output grid CSV: " << filepath << "\n";
        return false;
    }

    file << "band_name,ix,iy,resolution,point_count,elevation_mean,elevation_min,elevation_max,semantic_class,confidence,traversability\n";
    for (const auto& c : cells) {
        file << c.band_name << ","
             << c.ix << ","
             << c.iy << ","
             << std::fixed << std::setprecision(4) << c.resolution << ","
             << c.point_count << ","
             << std::setprecision(5) << c.elevation_mean << ","
             << std::setprecision(5) << c.elevation_min << ","
             << std::setprecision(5) << c.elevation_max << ","
             << static_cast<int>(c.semantic_class) << ","
             << std::setprecision(5) << c.confidence << ","
             << std::setprecision(4) << c.traversability << "\n";
    }
    return true;
}

} // namespace foveated_mapping
