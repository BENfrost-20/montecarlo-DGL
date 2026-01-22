/**
 * @file benchmarks.cpp
 * @brief Implementation of shared benchmark utilities and global configuration
 * 
 * @details Provides:
 * - Global configuration: sample counts and thread count
 * - Result serialization: save benchmark results to CSV files
 * - Shared infrastructure for all benchmark types
 * 
 * **Benchmark Components:**
 * - Integration benchmarks: src/apps/benchmarks/integration_benchmarks.cpp
 *   Tests convergence and accuracy of MC, IS, MCMC methods
 * - PSO benchmarks: src/apps/benchmarks/pso_benchmarks.cpp
 *   Performance testing on sphere, Rastrigin, Rosenbrock functions
 * - GA benchmarks: src/apps/benchmarks/ga_benchmarks.cpp
 *   Genetic algorithm with population visualization
 * 
 * **Sample Configuration:**
 * - Logarithmic scale: 10k, 50k, 100k, 500k, 1M samples
 * - Enables convergence analysis: E[error] ∝ 1/√N
 * 
 * @see benchmarks.hpp, saveResults()
 */

#include "apps/benchmarks.hpp"

/// Global sample count vector used across all benchmarks for convergence testing
const std::vector<size_t> n_samples_vector = {10'000, 50'000, 100'000, 500'000, 1'000'000};

/// Global OpenMP thread count (set at runtime)
unsigned int n_threads;

/**
 * @brief Export benchmark results to CSV file for analysis
 * @param filename Output CSV filename
 * @param results Vector of benchmark results {n_samples, integration_result, duration_ms}
 * @param function_expr String description of the integrated function
 * 
 * @details CSV Format:
 * - Header: Function description and column labels
 * - Each row: n_samples \\t result \\t duration_ms
 * 
 * Useful for:
 * - Plotting convergence curves: error vs sample count
 * - Computing empirical convergence rate
 * - Analyzing computational efficiency
 * - Comparing algorithm scalability
 */
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

