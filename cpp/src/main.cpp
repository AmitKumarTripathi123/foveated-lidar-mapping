#include "foveated_grid.hpp"
#include <iostream>
#include <chrono>
#include <string>

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --input <input_points.csv> --output <output_grid.csv> [--benchmark <iterations>]\n";
}

int main(int argc, char* argv[]) {
    std::string input_path = "tests/data/phase5_golden_input.csv";
    std::string output_path = "tests/output/cpp_grid.csv";
    int benchmark_iters = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--benchmark" && i + 1 < argc) {
            benchmark_iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    std::cout << "========================================================\n";
    std::cout << "  C++ Foveated 2.5D Spatial Grid Engine (Phase 5)\n";
    std::cout << "========================================================\n";
    std::cout << "Input CSV:  " << input_path << "\n";
    std::cout << "Output CSV: " << output_path << "\n";

    foveated_mapping::FoveatedGridEngine engine;
    auto points = engine.load_points_csv(input_path);
    std::cout << "Loaded " << points.size() << " classified points.\n";

    if (points.empty()) {
        std::cerr << "Warning: Loaded 0 points from " << input_path << "\n";
    }

    // Execute build_grid
    auto start_time = std::chrono::high_resolution_clock::now();
    auto cells = engine.build_grid(points);
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout << "Generated " << cells.size() << " 2.5D grid cells in " << elapsed_ms << " ms.\n";

    if (!engine.export_grid_csv(cells, output_path)) {
        std::cerr << "Error writing output CSV to " << output_path << "\n";
        return 1;
    }
    std::cout << "Successfully exported grid to " << output_path << "\n";

    // Benchmark mode if requested
    if (benchmark_iters > 0) {
        std::cout << "\nRunning C++ Benchmark across " << benchmark_iters << " iterations...\n";
        double total_ms = 0.0;
        for (int it = 0; it < benchmark_iters; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto c = engine.build_grid(points);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        double avg_ms = total_ms / benchmark_iters;
        double fps = 1000.0 / avg_ms;
        std::cout << "--------------------------------------------------------\n";
        std::cout << "Mean Processing Latency: " << avg_ms << " ms\n";
        std::cout << "Throughput:              " << fps << " FPS\n";
        std::cout << "Point Throughput:        " << (points.size() / (avg_ms / 1000.0)) << " points/sec\n";
        std::cout << "--------------------------------------------------------\n";
    }

    return 0;
}
