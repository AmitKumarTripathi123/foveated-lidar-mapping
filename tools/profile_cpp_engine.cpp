#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <string>
#include <random>
#include <iomanip>

struct Point { float x, y, z, intensity; uint8_t class_id; float confidence; };

int main() {
    size_t N = 100000;
    std::vector<Point> points(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_xy(-70.0f, 70.0f);
    std::uniform_real_distribution<float> dist_z(-3.0f, 5.0f);
    for (size_t i = 0; i < N; ++i) {
        points[i] = {dist_xy(rng), dist_xy(rng), dist_z(rng), 0.5f, 0, 0.9f};
    }

    std::cout << "Profiling 100,000 LiDAR Points Breakdown...\n";

    // 1. Distance with std::sqrt vs squared distance
    auto t0 = std::chrono::high_resolution_clock::now();
    double sum_r = 0;
    for (size_t i = 0; i < N; ++i) {
        float r = std::sqrt(points[i].x * points[i].x + points[i].y * points[i].y);
        sum_r += r;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sqrt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    double sum_r2 = 0;
    for (size_t i = 0; i < N; ++i) {
        float r2 = points[i].x * points[i].x + points[i].y * points[i].y;
        sum_r2 += r2;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double r2_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Cell coordinate math: division vs multiplication with inverse
    float res = 0.05f;
    float inv_res = 1.0f / res;
    t0 = std::chrono::high_resolution_clock::now();
    int64_t sum_ix = 0;
    for (size_t i = 0; i < N; ++i) {
        int64_t ix = static_cast<int64_t>(std::floor(points[i].x / res));
        sum_ix += ix;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double div_floor_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    int64_t sum_ix2 = 0;
    for (size_t i = 0; i < N; ++i) {
        int64_t ix = static_cast<int64_t>(std::floor(points[i].x * inv_res));
        sum_ix2 += ix;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double mul_floor_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 3. String key generation vs 64-bit integer key generation
    t0 = std::chrono::high_resolution_clock::now();
    size_t str_len_sum = 0;
    for (size_t i = 0; i < N; ++i) {
        std::string key = "near_field:" + std::to_string(static_cast<int>(points[i].x)) + ":" + std::to_string(static_cast<int>(points[i].y));
        str_len_sum += key.size();
    }
    t1 = std::chrono::high_resolution_clock::now();
    double str_key_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    uint64_t int_key_sum = 0;
    for (size_t i = 0; i < N; ++i) {
        int32_t ix = static_cast<int32_t>(points[i].x);
        int32_t iy = static_cast<int32_t>(points[i].y);
        uint64_t key = (static_cast<uint64_t>(0) << 48) | ((static_cast<uint64_t>(ix + 50000) & 0xFFFFFF) << 24) | (static_cast<uint64_t>(iy + 50000) & 0xFFFFFF);
        int_key_sum += key;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double int_key_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 4. Hash Map Insertion: std::string map vs uint64_t map
    t0 = std::chrono::high_resolution_clock::now();
    std::unordered_map<std::string, float> str_map;
    str_map.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        std::string key = "near_field:" + std::to_string(static_cast<int>(points[i].x)) + ":" + std::to_string(static_cast<int>(points[i].y));
        str_map[key] += points[i].z;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double str_map_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    std::unordered_map<uint64_t, float> int_map;
    int_map.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        int32_t ix = static_cast<int32_t>(points[i].x);
        int32_t iy = static_cast<int32_t>(points[i].y);
        uint64_t key = (static_cast<uint64_t>(0) << 48) | ((static_cast<uint64_t>(ix + 50000) & 0xFFFFFF) << 24) | (static_cast<uint64_t>(iy + 50000) & 0xFFFFFF);
        int_map[key] += points[i].z;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double int_map_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "========================================================\n";
    std::cout << "  C++ MICRO-PROFILING RESULTS (100,000 points)\n";
    std::cout << "========================================================\n";
    std::cout << "Distance sqrt():           " << sqrt_ms << " ms\n";
    std::cout << "Distance r^2 (No sqrt):    " << r2_ms << " ms  (" << (sqrt_ms/r2_ms) << "x speedup)\n";
    std::cout << "Coordinate floor (div):    " << div_floor_ms << " ms\n";
    std::cout << "Coordinate floor (mul):    " << mul_floor_ms << " ms  (" << (div_floor_ms/mul_floor_ms) << "x speedup)\n";
    std::cout << "String Key Generation:     " << str_key_ms << " ms\n";
    std::cout << "Packed 64-bit Int Key:     " << int_key_ms << " ms  (" << (str_key_ms/int_key_ms) << "x speedup)\n";
    std::cout << "String Map Insertion:      " << str_map_ms << " ms\n";
    std::cout << "Int64 Map Insertion:       " << int_map_ms << " ms  (" << (str_map_ms/int_map_ms) << "x speedup)\n";
    std::cout << "========================================================\n";

    return 0;
}
