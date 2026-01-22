/**
 * @file benchmarks.hpp
 * @brief Benchmarking framework for Monte Carlo integration algorithms
 * @author Domenico
 * @date 12/3/25
 * 
 * @details Provides infrastructure for evaluating integration accuracy and performance
 * across multiple algorithms (uniform MC, importance sampling, MCMC) and dimensionalities.
 * 
 * **Key Features:**
 * - Multiple test domains: 1D, circle, sphere, rectangle, cylinder, parallelepiped, 5D-12D
 * - Configurable sample counts and thread counts
 * - Results export to CSV for analysis
 * - Gnuplot visualization support
 * - Both hardcoded integrands and parser-based expression support
 * 
 * **Benchmark Types:**
 * - Integration benchmarks: convergence and accuracy across sample sizes
 * - MH-specific benchmarks: burn-in, thinning, and proposal parameter effects
 * - Optimizer benchmarks: GA and PSO performance (in benchmarks/ subdirectory)
 */

#ifndef MONTECARLO_1_BENCHMARKS_HPP
#define MONTECARLO_1_BENCHMARKS_HPP

#include <montecarlo/domains/hypercylinder.hpp>
#include <montecarlo/domains/hyperrectangle.hpp>
#include <montecarlo/domains/hypersphere.hpp>
#include <montecarlo/geometry.hpp>
#include <montecarlo/integrators/MCintegrator.hpp>
#include <montecarlo/integrators/MHintegrator.hpp>
#include <montecarlo/integrators/ISintegrator.hpp>
#include <montecarlo/proposals/uniformProposal.hpp>
#include <montecarlo/utils/plotter.hpp>
#include <montecarlo/optimizers/PSO.hpp>
#include <montecarlo/optimizers/GA.hpp>

#include <fstream>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <cmath>
#include <cstdint>

/// Global benchmark configuration
extern const std::vector<size_t> n_samples_vector;
extern unsigned int n_threads;

/**
 * @struct results
 * @brief Stores benchmark result for a single sample count.
 * @details Contains the number of samples, estimated integral, and execution time.
 */
struct results {
    size_t n_samples;
    std::string integration_result;
    std::string duration;
};

/**
 * @brief Save benchmark results to a CSV file.
 * @param filename Output CSV filename
 * @param results Vector of result structures
 * @param function_expr String describing the integrand for reference
 */
void saveResults(const std::string &filename, const std::vector<results> &results, const std::string &function_expr);

// Functions that contain both the domains and the functions to integrate over them

/**
 * @brief 1D integration test: f(x) = x²
 * @details Domain: [0,1], Expected: 1/3 ≈ 0.333...
 */
void uniDimIntegration();

/**
 * @brief 2D circular integration: f(x,y) = x²+y² over unit disk
 */
void circleIntegration();

/**
 * @brief 3D spherical integration: f(x,y,z) = x²+y²+z² over unit ball
 */
void sphereIntegration();

/**
 * @brief 2D rectangular integration over axis-aligned box
 */
void rectangularIntegration();

/**
 * @brief 3D cylindrical integration: hypercylinder base with height
 */
void cylinderIntegration();

/**
 * @brief 3D parallelepiped (rectangular box) integration
 */
void parallelepipedIntegration();

/**
 * @brief 5-dimensional integration benchmark
 */
void fiveDimIntegration();

/**
 * @brief 4-dimensional integration benchmark
 */
void fourDimIntegration();

/**
 * @brief 8-dimensional integration benchmark
 */
void eightDimIntegration();

/**
 * @brief 12-dimensional integration benchmark
 */
void twelveDimIntegration();

// --- Monte Carlo Integration Benchmarks ---
// (Implemented in benchmarks/integration_benchmarks.cpp)

/**
 * @brief Run integration benchmarks with hardcoded integrands.
 * @param useGnuplot If true, generate gnuplot script for visualization.
 */
void runBenchmarks(bool useGnuplot);

/**
 * @brief Run integration benchmarks with custom mathematical expression.
 * @param expression Mathematical function string (muParserX syntax)
 * @param useGnuplot If true, generate gnuplot script
 */
void runBenchmarks(const std::string& expression, bool useGnuplot);

/**
 * @brief Run Metropolis-Hastings MCMC integration benchmarks.
 */
void runBenchmarksMH();

// --- PSO Optimization Benchmarks ---
// (Implemented in benchmarks/pso_benchmarks.cpp)
void runOptimizationBenchmarksPSO();

// --- GA Optimization Benchmarks ---
// (Implemented in benchmarks/ga_benchmarks.cpp)
void runOptimizationBenchmarksGA();


#endif //MONTECARLO_1_BENCHMARKS_HPP