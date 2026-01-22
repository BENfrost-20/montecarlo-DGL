#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>

#include "montecarlo/rng/rng_global.hpp"
#include "montecarlo/rng/rng_factory.hpp"
#include "montecarlo/optimizers/PSO.hpp"
#include "montecarlo/optimizers/GA.hpp"

double rosenbrock(const std::vector<double>& x) {
    double a = 1.0, b = 100.0;
    return std::pow(a - x[0], 2) + b * std::pow(x[1] - x[0]*x[0], 2);
}

double run_pso_test() {
    mc::optim::PSOConfig cfg;
    cfg.population_size = 20;
    cfg.max_iterations = 30;
    cfg.inertia_weight = 0.7;
    cfg.cognitive_coeff = 1.5;
    cfg.social_coeff = 1.5;

    mc::optim::PSO pso(cfg);
    pso.setObjectiveFunction(rosenbrock);
    pso.setBounds({-5.0, -5.0}, {5.0, 5.0});
    pso.setMode(mc::optim::OptimizationMode::MINIMIZE);

    mc::optim::Solution best = pso.optimize();
    return best.value;
}

double run_ga_test() {
    mc::optim::GAConfig cfg;
    cfg.population_size = 20;
    cfg.max_generations = 30;
    cfg.tournament_k = 3;
    cfg.crossover_rate = 0.9;
    cfg.mutation_rate = 0.1;
    cfg.mutation_sigma = 0.1;
    cfg.elitism_count = 1;

    mc::optim::GA ga(cfg);
    ga.setObjectiveFunction(rosenbrock);
    ga.setBounds({-5.0, -5.0}, {5.0, 5.0});
    ga.setMode(mc::optim::OptimizationMode::MINIMIZE);

    mc::optim::Solution best = ga.optimize();
    return best.value;
}

std::vector<double> generate_samples(int n_samples, std::uint64_t stream_id) {
    std::vector<double> results(n_samples);

    #pragma omp parallel for
    for (int i = 0; i < n_samples; ++i) {
        // Deterministic per-index stream; independent of scheduling
        auto rng = mc::rng::make_engine(stream_id + static_cast<std::uint64_t>(i));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        results[i] = dist(rng);
    }

    return results;
}

int main() {
    std::cout << "=== RNG Reproducibility Test ===\n\n";

    bool all_passed = true;

    // Seed can be set only once (library policy)
    std::cout << "Init: set global seed once...\n";
    bool seed_ok = mc::rng::set_global_seed(12345u);
    std::uint32_t seed_now = mc::rng::get_global_seed();

    std::cout << "  set_global_seed returned: " << (seed_ok ? "true" : "false") << "\n";
    std::cout << "  get_global_seed: " << seed_now << "\n\n";

    if (!seed_ok || seed_now != 12345u) {
        std::cout << "  FAILED: Global seed initialization failed\n\n";
        return 1;
    }

    // =========================================================================
    // Test 1: PSO determinism within same process (same global seed)
    // =========================================================================
    std::cout << "Test 1: PSO determinism...\n";
    double pso_run1 = run_pso_test();
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
    // Test 2: GA determinism within same process (same global seed)
    // =========================================================================
    std::cout << "Test 2: GA determinism...\n";
    double ga_run1 = run_ga_test();
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
    // Test 3: Parallel sample generation determinism
    // =========================================================================
    std::cout << "Test 3: Parallel sample generation reproducibility...\n";

    int n_samples = 1000;
    auto samples1 = generate_samples(n_samples, 9000ULL);
    auto samples2 = generate_samples(n_samples, 9000ULL);

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
    // Summary
    // =========================================================================
    std::cout << "=================================\n";
    if (all_passed) {
        std::cout << "ALL TESTS PASSED!\n";
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED!\n";
        return 1;
    }
}