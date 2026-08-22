#include "foveated_grid.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>

namespace foveated_mapping {

namespace {

// Internal accumulator struct for cell point aggregation
struct CellAccumulator {
    int64_t ix{0};
    int64_t iy{0};
    int16_t band_idx{0};
    int64_t count{0};
    double sum_z{0.0};
    float min_z{std::numeric_limits<float>::infinity()};
    float max_z{-std::numeric_limits<float>::infinity()};
    double sum_conf{0.0};
    uint8_t best_class{SuperClass::IGNORE_LABEL};
    int best_priority{-1};

    inline void add_point(float z, uint8_t class_id, float conf, int priority) {
        count++;
        sum_z += z;
        if (z < min_z) min_z = z;
        if (z > max_z) max_z = z;
        sum_conf += conf;

        if (priority > best_priority) {
            best_priority = priority;
            best_class = class_id;
        }
    }
};

// High-Performance Flat Spatial Hash Grid with Linear Probing
class FlatSpatialGrid {
public:
    struct Entry {
        uint64_t key{0}; // 0 indicates empty slot
        CellAccumulator acc;
    };

    explicit FlatSpatialGrid(size_t expected_points) {
        // Find next power of 2 for capacity >= expected_points * 2
        size_t cap = 16384;
        while (cap < (expected_points * 2) && cap < (1ULL << 24)) {
            cap <<= 1;
        }
        capacity_ = cap;
        mask_ = capacity_ - 1;
        entries_.resize(capacity_);
        active_indices_.reserve(std::min(expected_points, capacity_ / 2));
    }

    inline void insert_point(
        uint64_t key,
        int16_t band_idx,
        int64_t ix,
        int64_t iy,
        float z,
        uint8_t class_id,
        float conf,
        int priority
    ) {
        // Fast 64-bit integer hash mixer
        uint64_t h = key;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        size_t idx = h & mask_;

        while (true) {
            if (entries_[idx].key == key) {
                // Existing cell hit
                entries_[idx].acc.add_point(z, class_id, conf, priority);
                return;
            }
            if (entries_[idx].key == 0) {
                // New cell allocation
                entries_[idx].key = key;
                auto& acc = entries_[idx].acc;
                acc.band_idx = band_idx;
                acc.ix = ix;
                acc.iy = iy;
                acc.count = 1;
                acc.sum_z = z;
                acc.min_z = z;
                acc.max_z = z;
                acc.sum_conf = conf;
                acc.best_class = class_id;
                acc.best_priority = priority;
                active_indices_.push_back(idx);
                return;
            }
            idx = (idx + 1) & mask_;
        }
    }

    const std::vector<size_t>& active_indices() const { return active_indices_; }
    const Entry& get_entry(size_t idx) const { return entries_[idx]; }

private:
    size_t capacity_{16384};
    size_t mask_{16383};
    std::vector<Entry> entries_;
    std::vector<size_t> active_indices_;
};

} // anonymous namespace

FoveatedGridEngine::FoveatedGridEngine() {
    bands_ = {
        {"near_field",     0.0f,  10.0f, 0.05f},
        {"mid_near_field", 10.0f, 30.0f, 0.10f},
        {"mid_far_field",  30.0f, 60.0f, 0.25f},
        {"far_field",      60.0f, 100.0f, 0.50f}
    };
    init_internal_bands();
}

FoveatedGridEngine::FoveatedGridEngine(const std::vector<FoveationBand>& custom_bands)
    : bands_(custom_bands) {
    init_internal_bands();
}

void FoveatedGridEngine::init_internal_bands() {
    fast_bands_.clear();
    fast_bands_.reserve(bands_.size());
    for (size_t i = 0; i < bands_.size(); ++i) {
        const auto& b = bands_[i];
        InternalBand ib;
        ib.name = b.name;
        ib.min_range = b.min_range;
        ib.max_range = b.max_range;
        ib.min_range_sq = b.min_range * b.min_range;
        ib.max_range_sq = b.max_range * b.max_range;
        ib.voxel_size = b.voxel_size;
        ib.inv_voxel_size = 1.0f / b.voxel_size;
        ib.band_idx = static_cast<int16_t>(i);
        fast_bands_.push_back(ib);
    }
}

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

std::vector<GridCell> FoveatedGridEngine::build_grid_raw(
    const float* pts_ptr,
    size_t num_points,
    size_t feat_dim,
    const int64_t* lbls_ptr,
    const float* confs_ptr
) const {
    if (num_points == 0 || pts_ptr == nullptr) {
        return {};
    }

    FlatSpatialGrid grid(num_points);
    const size_t num_bands = fast_bands_.size();

    // Hot Ingestion & Aggregation Loop
    for (size_t i = 0; i < num_points; ++i) {
        size_t offset = i * feat_dim;
        float x = pts_ptr[offset + 0];
        float y = pts_ptr[offset + 1];
        float z = pts_ptr[offset + 2];

        // Strict finiteness check
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            continue;
        }

        // Fast squared distance comparison (avoids expensive std::sqrt)
        float r2 = x * x + y * y;
        if (r2 < 0.0f || r2 >= 10000.0f) {
            continue; // Range [0.0, 100.0) m -> [0.0, 10000.0) m^2
        }

        // Find band
        int matched_band_idx = -1;
        for (size_t b = 0; b < num_bands; ++b) {
            if (r2 >= fast_bands_[b].min_range_sq && r2 < fast_bands_[b].max_range_sq) {
                matched_band_idx = static_cast<int>(b);
                break;
            }
        }

        if (matched_band_idx < 0) {
            continue;
        }

        const auto& band = fast_bands_[matched_band_idx];

        // Coordinate indexing using precomputed inverse resolution multiplication
        int64_t ix = static_cast<int64_t>(std::floor(x * band.inv_voxel_size));
        int64_t iy = static_cast<int64_t>(std::floor(y * band.inv_voxel_size));

        // 64-bit packed integer key: (band_idx + 1) << 56 | (ix + 100000) << 28 | (iy + 100000)
        uint64_t key = (static_cast<uint64_t>(matched_band_idx + 1) << 56) |
                       ((static_cast<uint64_t>(ix + 100000) & 0x0FFFFFFFULL) << 28) |
                       (static_cast<uint64_t>(iy + 100000) & 0x0FFFFFFFULL);

        uint8_t class_id = lbls_ptr ? static_cast<uint8_t>(lbls_ptr[i] & 0xFF) : SuperClass::IGNORE_LABEL;
        float conf = confs_ptr ? confs_ptr[i] : 1.0f;
        int priority = get_semantic_priority(class_id);

        grid.insert_point(key, matched_band_idx, ix, iy, z, class_id, conf, priority);
    }

    const auto& active_idx = grid.active_indices();
    std::vector<GridCell> result;
    result.reserve(active_idx.size());

    for (size_t idx : active_idx) {
        const auto& acc = grid.get_entry(idx).acc;
        const auto& band = fast_bands_[acc.band_idx];

        GridCell cell;
        cell.band_name = band.name;
        cell.ix = acc.ix;
        cell.iy = acc.iy;
        cell.resolution = band.voxel_size;
        cell.point_count = acc.count;
        cell.elevation_mean = (acc.count > 0) ? static_cast<float>(acc.sum_z / acc.count) : 0.0f;
        cell.elevation_min = acc.min_z;
        cell.elevation_max = acc.max_z;
        cell.semantic_class = acc.best_class;
        cell.confidence = (acc.count > 0) ? static_cast<float>(acc.sum_conf / acc.count) : 0.0f;
        cell.traversability = calculate_traversability(acc.best_class);
        result.push_back(cell);
    }

    // Deterministic sorting matching Python reference: (band_name, iy, ix)
    std::sort(result.begin(), result.end(), [](const GridCell& a, const GridCell& b) {
        if (a.band_name != b.band_name) return a.band_name < b.band_name;
        if (a.iy != b.iy) return a.iy < b.iy;
        return a.ix < b.ix;
    });

    return result;
}

std::vector<GridCell> FoveatedGridEngine::build_grid(const std::vector<ClassifiedPoint>& points) const {
    if (points.empty()) return {};
    size_t N = points.size();

    // Flatten points to contiguous float buffer (feat_dim = 6)
    std::vector<float> pts_buf(N * 4);
    std::vector<int64_t> lbls_buf(N);
    std::vector<float> confs_buf(N);

    for (size_t i = 0; i < N; ++i) {
        pts_buf[i * 4 + 0] = points[i].x;
        pts_buf[i * 4 + 1] = points[i].y;
        pts_buf[i * 4 + 2] = points[i].z;
        pts_buf[i * 4 + 3] = points[i].intensity;
        lbls_buf[i] = points[i].class_id;
        confs_buf[i] = points[i].confidence;
    }

    return build_grid_raw(pts_buf.data(), N, 4, lbls_buf.data(), confs_buf.data());
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
