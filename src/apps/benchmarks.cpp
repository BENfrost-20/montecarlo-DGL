//
// Created by domenico on 12/3/25.
//
// Shared utilities and global configuration for all benchmarks.
// Specific benchmark implementations are located in:
//   - benchmarks/integration_benchmarks.cpp (Monte Carlo integration)
//   - benchmarks/pso_benchmarks.cpp (Particle Swarm Optimization)
//   - benchmarks/ga_benchmarks.cpp (Genetic Algorithm)
//

#include "apps/benchmarks.hpp"

// --- Global Configuration ---
const std::vector<size_t> n_samples_vector = {10'000, 50'000, 100'000, 500'000, 1'000'000};
unsigned int n_threads;

// --- Utility Functions ---

void saveResults(const std::string &filename, const std::vector<results> &results, const std::string &function_expr) {
    std::ofstream outfile;
    outfile.open(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to create results file " << filename << std::endl;
        return;
    }

    // Header: Function description and columns
    outfile << "Function: " << function_expr << "\n";
    outfile << "Number of points\tIntegration Result\tDuration (ms)\n";

    for (const auto &result : results) {
        outfile << result.n_samples << "\t"
                << result.integration_result << "\t"
                << result.duration << "\n";
    }
    outfile.close();
}

