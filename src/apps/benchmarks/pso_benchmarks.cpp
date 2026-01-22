//
// PSO (Particle Swarm Optimization) benchmarks
//

#include "apps/benchmarks.hpp"
#include <cmath>
#include <cstdint>

namespace opt = mc::optim;

/**
 * @brief Test 1: Sphere Function
 * Objective: Minimize f(x, y) = x^2 + y^2
 * Global Minimum: 0 at [0, 0]
 */
void runSphereTest(opt::PSO& pso, const opt::Coordinates& lower, const opt::Coordinates& upper) {
    std::cout << "Optimization Problem: Minimize Sphere Function in 2D" << std::endl;
    std::cout << "Search Space: [-10, 10] per dimension" << std::endl;
    std::cout << "Running optimizer..." << std::endl;

    // Define the objective function: Sphere Function f(x) = sum(x_i^2)
    opt::ObjectiveFunction sphere_function = [](const opt::Coordinates& coords) {
        opt::Real sum = 0.0;
        for (auto val : coords) {
            sum += val * val;
        }
        return sum;
    };

    // Set up the optimizer
    pso.setBounds(lower, upper);
    pso.setObjectiveFunction(sphere_function);
    pso.setMode(opt::OptimizationMode::MINIMIZE);

    // Set a callback to print progress every 10 iterations
    pso.setCallback([](const opt::Solution& current_best, size_t iteration) {
        if (iteration == 0 || iteration % 10 == 0) {
            std::cout << "[Sphere Test | Step " << std::setw(3) << iteration << "] "
                      << "Best Value: " << std::scientific << std::setprecision(5)
                      << current_best.value << std::defaultfloat << std::endl;
        }
    });

    // Execute
    auto start = std::chrono::high_resolution_clock::now();
    opt::Solution best_sol = pso.optimize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Report
    std::cout << "\nOptimization Completed in " << duration.count() << " ms." << std::endl;
    std::cout << "Best Value Found: " << std::fixed << std::setprecision(10) << best_sol.value << std::endl;
    std::cout << "Best Position: [ ";
    for (auto val : best_sol.params) {
        std::cout << std::fixed << std::setprecision(5) << val << " ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Test 2: Boundary Constraint Test
 * Objective: Minimize f(x, y) = x + y
 * This function is a constant inclined plane. There is no local minimum inside
 * the domain (constant gradient). Particles must push to the extreme lower limit.
 * Expected Minimum: -20.0 at [-10, -10]
 */
void runBoundaryTest(opt::PSO& pso, const opt::Coordinates& lower, const opt::Coordinates& upper) {
    std::cout << "\n-------------------------------------------" << std::endl;
    std::cout << "TEST 2: Boundary Constraint Test (Linear Plane)" << std::endl;
    std::cout << "Objective: f(x,y) = x + y (Minimization)" << std::endl;
    std::cout << "Search Space: [-10, 10] per dimension" << std::endl;
    std::cout << "Expected Result: -20.0 at [-10.0, -10.0]" << std::endl;
    std::cout << "Running optimizer..." << std::endl;

    // Define the linear function
    opt::ObjectiveFunction plane_function = [](const opt::Coordinates& coords) {
        opt::Real sum = 0.0;
        for (auto val : coords) {
            sum += val;
        }
        return sum;
    };

    // Set up the optimizer (reusing the instance)
    pso.setBounds(lower, upper);
    pso.setObjectiveFunction(plane_function);
    // Mode is already MINIMIZE, but good practice to ensure
    pso.setMode(opt::OptimizationMode::MINIMIZE);

    // Callback to print progress
    pso.setCallback([](const opt::Solution& current_best, size_t iteration) {
        if (iteration % 20 == 0) {
            std::cout << "[Boundary Test | Step " << std::setw(3) << iteration << "] "
                      << "Val: " << std::fixed << std::setprecision(4) << current_best.value << std::endl;
        }
    });

    // Execute
    auto start = std::chrono::high_resolution_clock::now();
    opt::Solution best_sol = pso.optimize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Report
    std::cout << "Optimization Completed in " << duration.count() << " ms." << std::endl;
    std::cout << "Best Value Found: " << std::fixed << std::setprecision(5) << best_sol.value << std::endl;
    std::cout << "Best Position: [ ";
    for (auto val : best_sol.params) {
        std::cout << std::fixed << std::setprecision(5) << val << " ";
    }
    std::cout << "]" << std::endl;

    // Verification
    if (std::abs(best_sol.value - (-20.0)) < 1e-3) {
        std::cout << ">> SUCCESS: Boundary minimum found correctly!" << std::endl;
    } else {
        std::cout << ">> WARNING: Did not reach the exact boundary." << std::endl;
    }
}

/**
 * @brief Test 3: High-Dimensional Rastrigin Function
 * Objective: Minimize f(x) = 10n + sum(x_i^2 - 10cos(2*pi*x_i))
 * Domain: [-5.12, 5.12]
 * Global Minimum: 0.0 at x = [0, 0, ..., 0]
 *
 * Why it is hard:
 * This function creates a grid of local minima. In high dimensions (e.g., 10D),
 * simplistic optimizers get stuck in a local valley instead of finding the global 0.
 * A successful run requires a good balance of exploration and exploitation.
 */
void runRastriginTest(opt::PSO& pso, int dim) {
    std::cout << "\n-------------------------------------------" << std::endl;
    std::cout << "TEST 3: High-Dimensional Stress Test (Rastrigin Function)" << std::endl;
    std::cout << "Dimension: " << dim << "D" << std::endl;
    std::cout << "Search Space: [-5.12, 5.12] per dimension" << std::endl;
    std::cout << "Goal: Find global minimum 0.0 (avoiding local traps)" << std::endl;

    // 1. Define the Rastrigin function
    opt::ObjectiveFunction rastrigin_func = [dim](const opt::Coordinates& coords) {
        double sum = 0.0;
        double A = 10.0;
        // M_PI is standard in <cmath>, if missing use 3.14159265358979323846
        double pi = 3.14159265358979323846;

        for (auto x : coords) {
            sum += (x * x) - (A * std::cos(2.0 * pi * x));
        }
        return A * dim + sum;
    };

    // 2. Define Bounds for N dimensions
    opt::Coordinates lower(dim, -5.12);
    opt::Coordinates upper(dim, 5.12);

    // 3. Configure PSO specifically for a harder problem
    // We need more particles and more time to explore 10 dimensions
    opt::PSOConfig hard_config;
    hard_config.population_size = 500; // Increased from 50 - with population 100 only finds local minima
    hard_config.max_iterations = 1000; // Increased from 100
    hard_config.inertia_weight = 0.729; // Classic "constriction factor" value
    hard_config.cognitive_coeff = 1.49;
    hard_config.social_coeff = 1.49;

    // We create a new local instance to not mess up the previous config
    opt::PSO local_pso(hard_config);

    local_pso.setBounds(lower, upper);
    local_pso.setObjectiveFunction(rastrigin_func);
    local_pso.setMode(opt::OptimizationMode::MINIMIZE);

    // Minimal callback to show we are alive
    local_pso.setCallback([](const opt::Solution& sol, size_t i) {
        if (i % 100 == 0) {
            std::cout << "[Rastrigin " << i << "] Best: "
                      << std::scientific << std::setprecision(4) << sol.value
                      << std::defaultfloat << std::endl;
        }
    });

    std::cout << "Running optimizer (this might take longer)..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    opt::Solution best_sol = local_pso.optimize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Optimization Completed in " << duration.count() << " ms." << std::endl;
    std::cout << "Best Value Found: " << std::fixed << std::setprecision(5) << best_sol.value << std::endl;

    // Validation: It's very hard to get exactly 0.0 in 10D without huge resources.
    // Anything below 1.0 is considered a "good" result for a basic PSO.
    // Anything close to 0.0 is excellent.
    if (best_sol.value < 1e-2) {
        std::cout << ">> SUCCESS: Global minimum found!" << std::endl;
    } else if (best_sol.value < 5.0) {
        std::cout << ">> ACCEPTABLE: Found a good local minimum, but not global." << std::endl;
    } else {
        std::cout << ">> FAIL: Stuck in a high local minimum." << std::endl;
    }
}

void runVisualPSOBenchmark() {
    std::cout << "===========================================" << std::endl;
    std::cout << "   Visual PSO Benchmark (2D Animation)" << std::endl;
    std::cout << "===========================================" << std::endl;

    // 1. Configuration
    opt::PSOConfig config;
    config.population_size = 60; // 30 particles
    config.max_iterations = 100;  // 50 frames for animation
    opt::PSO pso(config);

    // 2. Objective Function (Rastrigin 2D)
    // Global minimum is at (0,0)
    auto rastrigin = [](const opt::Coordinates& x) {
        double A = 10.0;
        double sum = 0.0;
        // M_PI is typically defined in <cmath>, ensure it is available or use 3.14...
        double pi = 3.14159265358979323846;
        for (double val : x) sum += val*val - A*std::cos(2*pi*val);
        return 2*A + sum;
    };

    pso.setObjectiveFunction(rastrigin);
    pso.setBounds({-5.12, -5.12}, {5.12, 5.12});
    pso.setMode(opt::OptimizationMode::MINIMIZE);

    // 3. Prepare Plotting
    std::string baseName = "pso_vis";
    std::string gridFile = "pso_grid.dat";

    std::cout << "Generating background grid (heatmap)..." << std::endl;
    // Save the static background (function landscape)
    mc::utils::saveFunctionGrid(gridFile, rastrigin, -5.12, 5.12, -5.12, 5.12, 100);

    // 4. Set Callback to save each frame
    pso.setCallback([&](const opt::Solution&, size_t iter) {
        // Use the public getter to access particle positions
        mc::utils::saveSwarmFrame(baseName, iter, pso.getParticles());
        std::cout << "Saved frame " << iter << "/" << config.max_iterations << "\r" << std::flush;
    });

    // 5. Run Optimization
    std::cout << "Running optimization..." << std::endl;
    pso.optimize();
    std::cout << "\nOptimization finished." << std::endl;

    // 6. Launch Animation
    std::cout << "Launching Gnuplot animation..." << std::endl;
    mc::utils::createPSOAnimationScript("run_pso.gp", gridFile, baseName, config.max_iterations, "PSO Rastrigin 2D");
}

void runVisualPSO3DBenchmark() {
    std::cout << "===========================================" << std::endl;
    std::cout << "   Visual PSO Benchmark (3D Animation)" << std::endl;
    std::cout << "===========================================" << std::endl;

    // 1. Configuration
    opt::PSOConfig config;
    config.population_size = 100;
    config.max_iterations = 150;
    opt::PSO pso(config);

    // 2. Objective: 3D Rastrigin Function
    auto rastrigin3D = [](const opt::Coordinates& x) {
        double sum = 0.0;
        double A = 10.0;
        double pi = 3.14159265358979323846;
        for (double val : x) {
            sum += val * val - A * std::cos(2 * pi * val);
        }
        return 3.0 * A + sum;
    };

    pso.setObjectiveFunction(rastrigin3D);

    // Bounds [-5.12, 5.12]
    double min_b = -5.12;
    double max_b = 5.12;
    pso.setBounds({min_b, min_b, min_b}, {max_b, max_b, max_b});
    pso.setMode(opt::OptimizationMode::MINIMIZE);

    // 3. Setup Visualization
    std::string baseName = "pso_vis_3d";
    std::string slicesFile = "pso_slices_3d.dat"; // [NEW] File for wall heatmaps

    // [NEW] Generate the 3D Slices (Walls)
    std::cout << "Generating 3D function slices (walls)..." << std::endl;
    mc::utils::saveFunctionSlices3D(slicesFile, rastrigin3D, min_b, max_b, 50); // 50 = resolution

    // 4. Callback
    pso.setCallback([&](const opt::Solution&, size_t iter) {
        mc::utils::saveSwarmFrame(baseName, iter, pso.getParticles());
        if (iter % 10 == 0) std::cout << "Generating Frame " << iter << "/" << config.max_iterations << "\r" << std::flush;
    });

    // 5. Run
    std::cout << "Running 3D optimization..." << std::endl;
    pso.optimize();
    std::cout << "\nOptimization finished." << std::endl;

    // 6. Launch 3D Animation (Pass slicesFile)
    std::cout << "Launching Gnuplot 3D animation..." << std::endl;
    mc::utils::createPSOAnimationScript3D("run_pso_3d.gp", slicesFile, baseName, config.max_iterations, "PSO 3D Rastrigin", min_b, max_b);
}

// --- Main Optimization Benchmark Entry Point ---

void runOptimizationBenchmarksPSO() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   Particle Swarm Optimization (PSO) Benchmark" << std::endl;
    std::cout << "=========================================" << std::endl;

    // 1. Configuration for PSO
    opt::PSOConfig config;
    config.population_size = 50;   // Number of particles
    config.max_iterations = 100;   // Iterations
    config.inertia_weight = 0.7;
    config.cognitive_coeff = 1.5;
    config.social_coeff = 1.5;

    // 2. Instantiate PSO
    opt::PSO pso(config);

    // 3. Define Shared Bounds [-10, 10]
    opt::Coordinates lower_bounds = {-10.0, -10.0};
    opt::Coordinates upper_bounds = {10.0, 10.0};

    try {
        // Run Test 1
        runSphereTest(pso, lower_bounds, upper_bounds);

        // Run Test 2
        runBoundaryTest(pso, lower_bounds, upper_bounds);

        // Run Test 3: 10-Dimensional Rastrigin
        runRastriginTest(pso, 10);

        // Run visual test:
        runVisualPSOBenchmark();

        // Run 3D visual test
        runVisualPSO3DBenchmark();

    } catch (const std::exception& e) {
        std::cerr << "Optimization failed: " << e.what() << std::endl;
    }
}
