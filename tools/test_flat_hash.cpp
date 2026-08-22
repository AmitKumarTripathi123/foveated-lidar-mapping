#include <iostream>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <random>
#include <cstdint>

struct Accumulator {
    int64_t ix{0};
    int64_t iy{0};
    int16_t band_idx{0};
    int64_t count{0};
    double sum_z{0.0};
    float min_z{1e9f};
    float max_z{-1e9f};
    double sum_conf{0.0};
    uint8_t best_class{255};
    int best_priority{-1};
};

// Flat open-addressing spatial hash table
class FlatSpatialGrid {
public:
    struct Entry {
        uint64_t key{0}; // 0 = empty
        Accumulator acc;
    };

    explicit FlatSpatialGrid(size_t capacity_pow2 = 131072) // 128K slots
        : mask_(capacity_pow2 - 1), entries_(capacity_pow2) {}

    inline void add_point(uint64_t key, int16_t band_idx, int64_t ix, int64_t iy, float z, uint8_t class_id, float conf, int priority) {
        // Fast Murmur-style 64-bit hash
        uint64_t h = key;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        size_t idx = h & mask_;

        while (true) {
            if (entries_[idx].key == key) {
                // Existing cell
                auto& acc = entries_[idx].acc;
                acc.count++;
                acc.sum_z += z;
                if (z < acc.min_z) acc.min_z = z;
                if (z > acc.max_z) acc.max_z = z;
                acc.sum_conf += conf;
                if (priority > acc.best_priority) {
                    acc.best_priority = priority;
                    acc.best_class = class_id;
                }
                return;
            }
            if (entries_[idx].key == 0) {
                // New cell
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
    size_t mask_;
    std::vector<Entry> entries_;
    std::vector<size_t> active_indices_;
};

int main() {
    size_t N = 100000;
    std::vector<uint64_t> keys(N);
    std::vector<float> z(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist_c(-1000, 1000);
    std::uniform_real_distribution<float> dist_z(-3.0f, 5.0f);
    for (size_t i = 0; i < N; ++i) {
        int ix = dist_c(rng);
        int iy = dist_c(rng);
        keys[i] = (static_cast<uint64_t>(0) << 48) | ((static_cast<uint64_t>(ix + 50000) & 0xFFFFFF) << 24) | (static_cast<uint64_t>(iy + 50000) & 0xFFFFFF);
        z[i] = dist_z(rng);
    }

    // std::unordered_map
    auto t0 = std::chrono::high_resolution_clock::now();
    std::unordered_map<uint64_t, Accumulator> map;
    map.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        auto& acc = map[keys[i]];
        acc.count++;
        acc.sum_z += z[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double unord_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // FlatSpatialGrid
    t0 = std::chrono::high_resolution_clock::now();
    FlatSpatialGrid flat(262144);
    for (size_t i = 0; i < N; ++i) {
        flat.add_point(keys[i], 0, 0, 0, z[i], 0, 0.9f, 1);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double flat_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "std::unordered_map<uint64_t, Acc>: " << unord_ms << " ms\n";
    std::cout << "FlatSpatialGrid (Open Addressing): " << flat_ms << " ms (" << (unord_ms/flat_ms) << "x faster!)\n";
    return 0;
}
