/**
 * @file test_rng_reproducibility.cpp
 * @brief Test to verify that the global RNG system provides deterministic results
 * @date 2026-01-21
 * 
 * This test demonstrates that:
 * 1. With the same global seed, two consecutive runs produce identical results
 * 2. Different seeds produce different (but still deterministic) results
 * 3. The system works correctly with OpenMP parallelization
 * 
 * Compile with:
 *   g++ -std=c++20 -fopenmp -O2 -I../src test_rng_reproducibility.cpp \
 *       ../build/src/montecarlo/rng/rng_global.o \
 *       ../build/src/montecarlo/rng/rng_factory.o \
 *       -o test_rng_reproducibility
 * 
 * Or via CMake (add as test target).
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <omp.h>

#include "montecarlo/rng/rng_global.hpp"
#include "montecarlo/rng/rng_factory.hpp"
#include "montecarlo/optimizers/PSO.hpp"
#include "montecarlo/optimizers/GA.hpp"

using namespace optimizers;

// Simple test function: Rosenbrock in 2D
double rosenbrock(const std::vector<double>& x) {
    double a = 1.0, b = 100.0;
    return std::pow(a - x[0], 2) + b * std::pow(x[1] - x[0]*x[0], 2);
}

// Run PSO and return best value
double run_pso_test() {
    PSOConfig cfg;
    cfg.population_size = 20;
    cfg.max_iterations = 30;
    cfg.inertia_weight = 0.7;
    cfg.cognitive_coeff = 1.5;
    cfg.social_coeff = 1.5;

    PSO pso(cfg);
    pso.setObjectiveFunction(rosenbrock);
    pso.setBounds({-5.0, -5.0}, {5.0, 5.0});
    pso.setMode(OptimizationMode::MINIMIZE);
    
    Solution best = pso.optimize();
    return best.value;
}

// Run GA and return best value
double run_ga_test() {
    GAConfig cfg;
    cfg.population_size = 20;
    cfg.max_generations = 30;
    cfg.tournament_k = 3;
    cfg.crossover_rate = 0.9;
    cfg.mutation_rate = 0.1;
    cfg.mutation_sigma = 0.1;
    cfg.elitism_count = 1;

    GA ga(cfg);
    ga.setObjectiveFunction(rosenbrock);
    ga.setBounds({-5.0, -5.0}, {5.0, 5.0});
    ga.setMode(OptimizationMode::MINIMIZE);
    
    Solution best = ga.optimize();
    return best.value;
}

// Generate some random samples with make_thread_engine
std::vector<double> generate_samples(int n_samples, std::uint64_t stream_id) {
    std::vector<double> results(n_samples);
    
    #pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
        auto rng = mc::make_thread_engine(stream_id + static_cast<std::uint64_t>(i));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        results[i] = dist(rng);
    }
    
    return results;
}

int main() {
    std::cout << "=== RNG Reproducibility Test ===\n\n";
    
    bool all_passed = true;
    
    // =========================================================================
    // Test 1: Same seed produces same PSO results
    // =========================================================================
    std::cout << "Test 1: PSO reproducibility with same seed...\n";
    
    mc::set_global_seed(12345u);
    double pso_run1 = run_pso_test();
    
    mc::set_global_seed(12345u);  // Reset to same seed
    double pso_run2 = run_pso_test();
    
    std::cout << "  Run 1: " << std::fixed << std::setprecision(10) << pso_run1 << "\n";
    std::cout << "  Run 2: " << std::fixed << std::setprecision(10) << pso_run2 << "\n";
    
    if (std::abs(pso_run1 - pso_run2) < 1e-12) {
        std::cout << "  PASSED: Results are identical\n\n";
    } else {
        std::cout << "  FAILED: Results differ!\n\n";
        all_passed = false;
    }
    
    // =========================================================================
    // Test 2: Same seed produces same GA results
    // =========================================================================
    std::cout << "Test 2: GA reproducibility with same seed...\n";
    
    mc::set_global_seed(54321u);
    double ga_run1 = run_ga_test();
    
    mc::set_global_seed(54321u);  // Reset to same seed
    double ga_run2 = run_ga_test();
    
    std::cout << "  Run 1: " << std::fixed << std::setprecision(10) << ga_run1 << "\n";
    std::cout << "  Run 2: " << std::fixed << std::setprecision(10) << ga_run2 << "\n";
    
    if (std::abs(ga_run1 - ga_run2) < 1e-12) {
        std::cout << "  PASSED: Results are identical\n\n";
    } else {
        std::cout << "  FAILED: Results differ!\n\n";
        all_passed = false;
    }
    
    // =========================================================================
    // Test 3: Different seeds produce different results
    // =========================================================================
    std::cout << "Test 3: Different seeds produce different results...\n";
    
    mc::set_global_seed(11111u);
    double pso_seed1 = run_pso_test();
    
    mc::set_global_seed(22222u);
    double pso_seed2 = run_pso_test();
    
    std::cout << "  Seed 11111: " << std::fixed << std::setprecision(10) << pso_seed1 << "\n";
    std::cout << "  Seed 22222: " << std::fixed << std::setprecision(10) << pso_seed2 << "\n";
    
    if (std::abs(pso_seed1 - pso_seed2) > 1e-10) {
        std::cout << "  PASSED: Results are different (as expected)\n\n";
    } else {
        std::cout << "  WARNING: Results are suspiciously similar\n\n";
        // Not a failure per se, but worth noting
    }
    
    // =========================================================================
    // Test 4: Parallel sample generation is deterministic
    // =========================================================================
    std::cout << "Test 4: Parallel sample generation reproducibility...\n";
    
    int n_samples = 1000;
    
    mc::set_global_seed(99999u);
    auto samples1 = generate_samples(n_samples, 0);
    
    mc::set_global_seed(99999u);  // Reset to same seed
    auto samples2 = generate_samples(n_samples, 0);
    
    int mismatches = 0;
    for (int i = 0; i < n_samples; ++i) {
        if (std::abs(samples1[i] - samples2[i]) > 1e-15) {
            mismatches++;
        }
    }
    
    std::cout << "  Generated " << n_samples << " samples in parallel\n";
    std::cout << "  Mismatches between runs: " << mismatches << "\n";
    
    if (mismatches == 0) {
        std::cout << "  PASSED: All samples are identical\n\n";
    } else {
        std::cout << "  FAILED: Some samples differ!\n\n";
        all_passed = false;
    }
    
    // =========================================================================
    // Test 5: get_global_seed returns the set value
    // =========================================================================
    std::cout << "Test 5: get_global_seed returns correct value...\n";
    
    mc::set_global_seed(42u);
    std::uint32_t retrieved = mc::get_global_seed();
    
    std::cout << "  Set seed: 42\n";
    std::cout << "  Got seed: " << retrieved << "\n";
    
    if (retrieved == 42u) {
        std::cout << "  PASSED: Seed getter works correctly\n\n";
    } else {
        std::cout << "  FAILED: Seed mismatch!\n\n";
        all_passed = false;
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "=================================\n";
    if (all_passed) {
        std::cout << "ALL TESTS PASSED!\n";
        std::cout << "The RNG system provides deterministic, reproducible results.\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
}
