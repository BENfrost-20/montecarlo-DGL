/**
 * @file integration_benchmarks.cpp
 * @brief Benchmark suite for Monte Carlo integration algorithms
 * 
 * @details Comprehensive evaluation of integration accuracy and performance across:
 * - Multiple algorithms: Uniform MC, Importance Sampling, MCMC
 * - Various dimensions: 1D through 12D
 * - Different domain types: rectangles, spheres, cylinders, hyperrectangles
 * - Parser-based and hardcoded function expressions
 * 
 * **Methodology:**
 * - Sweep across sample counts (logarithmic scale)
 * - Time execution of each integration
 * - Compute error relative to analytical reference when available
 * - Generate CSV output for statistical analysis
 * - Produce Gnuplot scripts for visualization
 * 
 * **Key Benchmarks:**
 * - uniDimIntegration(): ∫₀¹ x² dx = 1/3
 * - circleIntegration(): ∫ (x²+y²) dA over unit disk
 * - sphereIntegration(): ∫ (x²+y²+z²) dV over unit ball
 * - Higher-dimensional tests to study curse of dimensionality
 * 
 * @see runBenchmarks(), executeBenchmark(), saveResults()
 */

//
// Integration benchmarks for Monte Carlo methods
//

#include "apps/benchmarks.hpp"
#include <montecarlo/integrators/ISintegrator.hpp>
#include <montecarlo/utils/muParserXInterface.hpp>
#include <montecarlo/rng/rng_global.hpp>
#include <cmath>
#include <cstdint> 

// --- Helper for formatted console output ---

/**
 * @brief Generic execution loop for integration benchmarks.
 * @tparam dim Dimensionality of the integration domain.
 * @tparam Func Function type (integrand).
 * @param title Display title for the benchmark
 * @param filename Output CSV filename
 * @param integrator Integration engine to benchmark
 * @param domain Integration domain (used for visualization)
 * @param f Integrand function
 * @param useGnuplot If true, generate visualization script
 * @param rawDataFile Raw data filename for plotting
 * @param functionExpr String representation of the integrand
 * 
 * @details GENERIC execution loop.
 * Runs the integration for increasing sample sizes, measures time, plots results, and saves to file.
 */
template <size_t dim, typename Func>
void executeBenchmark(const std::string& title,
                      const std::string& filename,
                      mc::integrators::MontecarloIntegrator<dim>& integrator,
                      const mc::domains::IntegrationDomain<dim>& domain, // Needed for plotting
                      Func&& f,
                      bool useGnuplot,
                      const std::string& rawDataFile,
                      const std::string& functionExpr)      // Function string for display/save
{   
    // --- Helper for console output ---
    auto printLine = [&](std::size_t n,
                     const std::string& label,
                     double result,
                     long time_ms)
    {
        std::cout << std::setw(8)  << n << " | "
                << std::setw(30) << label << " | "
                << std::setw(20) << std::fixed << std::setprecision(6) << result << " | "
                << std::setw(5)  << time_ms << " ms\n";
    };

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << title << ":" << std::endl;
    std::cout << "Integrating Function: " << functionExpr << std::endl << std::endl;

    std::vector<results> testResults;

	mc::integrators::ISMontecarloIntegrator<dim> isIntegrator(domain);
    mc::proposals::UniformProposal<dim> uprop(domain);
    std::vector<double> init_mean(dim, 0.0);
    std::vector<double> init_sigma(dim, 2.5);
    auto bounds = domain.getBounds();
    for (size_t i = 0; i < dim; ++i) {
        init_mean[i]  = 0.5 * (bounds[i].first + bounds[i].second);
        init_sigma[i] = (bounds[i].second - bounds[i].first) / 3.0; // oppure /2.0
    }
    mc::proposals::GaussianProposal<dim> gprop(domain, init_mean, init_sigma);
    mc::proposals::MixtureProposal<dim> mix({&uprop, &gprop}, {0.5, 0.5});

    // Header table for console output
    std::cout << std::string(107, '-') << '\n';
    std::cout << std::setw(8)  << "Samples" << " | "
            << std::setw(30) << "Method"  << " | "
            << std::setw(20) << "Result"  << " | "
            << std::setw(6)  << "Time"    << '\n';
    std::cout << std::string(107, '-') << '\n';

    for (size_t n_i : n_samples_vector) {
        // 1. Normal MC (Data points are written to rawDataFile by the integrator)
        auto startTimer1 = std::chrono::high_resolution_clock::now();
        double result1 = integrator.OLDintegrate(f, n_i);
        auto endTimer1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer1 - startTimer1);

        // 2. Uniform IS (Data points are written to rawDataFile by the integrator)
        auto startTimer2 = std::chrono::high_resolution_clock::now();
        double result2 = isIntegrator.integrate(f, n_i, uprop, mc::rng::get_global_seed());
        auto endTimer2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer2 - startTimer2);

        // 3. Gaussian IS (Data points are written to rawDataFile by the integrator)
        auto startTimer3 = std::chrono::high_resolution_clock::now();
        double result3 = isIntegrator.integrate(f, n_i, gprop, mc::rng::get_global_seed());
        auto endTimer3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer3 - startTimer3);

        // 4. Mixture IS (Data points are written to rawDataFile by the integrator)
        auto startTimer4 = std::chrono::high_resolution_clock::now();
        double result4 = isIntegrator.integrate(f, n_i, mix, mc::rng::get_global_seed());
        auto endTimer4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(endTimer4 - startTimer4);

        // Store results
        results newLine;
        newLine.n_samples = n_i;
        newLine.integration_result = std::to_string(result1);
        newLine.duration = std::to_string(duration1.count());
        testResults.push_back(newLine); 

        // Console Output
        printLine(n_i, "Normal Sampling",              result1, duration1.count());
        printLine(n_i, "Uniform Importance Sampling",  result2, duration2.count());
        printLine(n_i, "Gaussian Importance Sampling", result3, duration3.count());
        printLine(n_i, "Mixture Importance Sampling",  result4, duration4.count());
        std::cout << std::string(107, '-') << '\n';

        // 2. Plotting (Inside the loop to show progress for each sample size)
        if (useGnuplot) {
            // Plot Domain Geometry (Inside/Outside points)
            mc::utils::createGnuplotScript(rawDataFile, domain, n_i);

            // Plot Function Value (f(x))
            mc::utils::createFunctionGnuplotScript(rawDataFile, domain, f, n_i);
        }
    }

    std::cout << "\nSaved txt file: " << filename << "\n";
    saveResults(filename, testResults, functionExpr);
}

// --- Domain-Specific Benchmark Wrappers ---

template <typename Func>
void runCircleBenchmark(Func f, const std::string& modeLabel, bool useGnuplot, const std::string& funcStr) {
    mc::domains::Hypersphere<2> circle(5.0);
    mc::integrators::MontecarloIntegrator<2> integrator(circle);
    std::string title = "2D Circle Integration (Radius 5) [" + modeLabel + "]";
    std::string filename = "resultsCircle_" + modeLabel + ".txt";
    std::string dataFile = "hsphere_samples.dat"; // File name must match what Integrator writes

    executeBenchmark(title, filename, integrator, circle, f, useGnuplot, dataFile, funcStr);
}

template <typename Func>
void runSphereBenchmark(Func f, const std::string& modeLabel, bool useGnuplot, const std::string& funcStr) {
    double radius = 10.0;
    mc::domains::Hypersphere<4> sphere(radius);
    mc::integrators::MontecarloIntegrator<4> integrator(sphere);
    std::string title = "4D Hypersphere Integration [" + modeLabel + "]";
    std::string filename = "resultsSphere4D_" + modeLabel + ".txt";
    std::string dataFile = "hsphere_samples.dat";

    executeBenchmark(title, filename, integrator, sphere, f, useGnuplot, dataFile, funcStr);
}

template <typename Func>
void runRectBenchmark(Func f, const std::string& modeLabel, bool useGnuplot, const std::string& funcStr) {
    std::array<double, 4> sides = {10.0, 5.0, 10.0, 5.0};
    mc::domains::HyperRectangle<4> rectangle(sides);
    mc::integrators::MontecarloIntegrator<4> integrator(rectangle);
    std::string title = "4D HyperRectangle Integration [" + modeLabel + "]";
    std::string filename = "resultsRectangle4D_" + modeLabel + ".txt";
    std::string dataFile = "hrectangle_samples.dat";

    executeBenchmark(title, filename, integrator, rectangle, f, useGnuplot, dataFile, funcStr);
}

template <typename Func>
void runCylinderBenchmark(Func f, const std::string& modeLabel, bool useGnuplot, const std::string& funcStr) {
    double radius = 5.0;
    double height = 10.0;
    mc::domains::HyperCylinder<4> cylinder(radius, height);
    mc::integrators::MontecarloIntegrator<4> integrator(cylinder);
    std::string title = "4D HyperCylinder Integration [" + modeLabel + "]";
    std::string filename = "resultsCylinder4D_" + modeLabel + ".txt";
    std::string dataFile = "cylinder_samples.dat";

    executeBenchmark(title, filename, integrator, cylinder, f, useGnuplot, dataFile, funcStr);
}


// --- Specific Implementations (Hardcoded vs Parser) ---

// 1. HARDCODED (C++ Lambda)
// Manually defining the function string description for the output file/console.

void circleIntegration(bool useGnuplot) {
    auto f = [](const mc::geom::Point<2> &x) { return x[0] * x[0] - x[1] * x[1]; };
    std::string funcStr = "x[0]^2 - x[1]^2";
    runCircleBenchmark(f, "Hardcoded", useGnuplot, funcStr);
}

void sphereIntegration(bool useGnuplot) {
    auto f = [](const mc::geom::Point<4> &x) { return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]; };
    std::string funcStr = "x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2";
    runSphereBenchmark(f, "Hardcoded", useGnuplot, funcStr);
}

void rectangularIntegration(bool useGnuplot) {
    auto f = [](const mc::geom::Point<4> &x) { return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]; };
    std::string funcStr = "x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2";
    runRectBenchmark(f, "Hardcoded", useGnuplot, funcStr);
}

void cylinderIntegration(bool useGnuplot) {
    auto f = [](const mc::geom::Point<4> &x) { return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]; };
    std::string funcStr = "x[0]^2 + x[1]^2 + x[2]^2 + x[3]^2";
    runCylinderBenchmark(f, "Hardcoded", useGnuplot, funcStr);
}

// 2. PARSER (muParserX)
// Using the 'expr' string directly read from file.

// 2. PARSER (muParserX) - THREAD SAFE WRAPPER

void circleIntegrationParser(const std::string& expr, bool useGnuplot) {
    using Parser = mc::utils::muParserXInterface<2, mc::geom::Point<2>>;
    Parser base(expr);

    auto f = [base](const mc::geom::Point<2>& x) -> double {
        thread_local Parser p = base;   // una copia per thread
        return p(x);
    };

    runCircleBenchmark(f, "Parser", useGnuplot, expr);
}

void sphereIntegrationParser(const std::string& expr, bool useGnuplot) {
    using Parser = mc::utils::muParserXInterface<4, mc::geom::Point<4>>;
    Parser base(expr);

    auto f = [base](const mc::geom::Point<4>& x) -> double {
        thread_local Parser p = base;
        return p(x);
    };

    runSphereBenchmark(f, "Parser", useGnuplot, expr);
}

void rectangularIntegrationParser(const std::string& expr, bool useGnuplot) {
    using Parser = mc::utils::muParserXInterface<4, mc::geom::Point<4>>;
    Parser base(expr);

    auto f = [base](const mc::geom::Point<4>& x) -> double {
        thread_local Parser p = base;
        return p(x);
    };

    runRectBenchmark(f, "Parser", useGnuplot, expr);
}

void cylinderIntegrationParser(const std::string& expr, bool useGnuplot) {
    using Parser = mc::utils::muParserXInterface<4, mc::geom::Point<4>>;
    Parser base(expr);

    auto f = [base](const mc::geom::Point<4>& x) -> double {
        thread_local Parser p = base;
        return p(x);
    };

    runCylinderBenchmark(f, "Parser", useGnuplot, expr);
}

// --- Main Entry Points ---

void runBenchmarks(bool useGnuplot) {
    n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 16;

    circleIntegration(useGnuplot);
    sphereIntegration(useGnuplot);
    rectangularIntegration(useGnuplot);
    cylinderIntegration(useGnuplot);
}

#include <cmath>
#include <iostream>
#include <functional>
#include <limits>
#include <random>
#include <string>

void runBenchmarksMH() {
    n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 16;

    constexpr double R = 10.0;
    mc::domains::Hypersphere<2> domain(R);

    std::function<double(const mc::geom::Point<2>&)> indicator =
        [&domain](const mc::geom::Point<2>& x) -> double {
            return domain.isInside(x) ? 1.0 : 0.0;
        };

    const double deviation = 0.15;
    const std::size_t burn_in = 20'000;
    const std::size_t thinning = 10;
    const std::size_t n_samples = 1'000'000;
    const std::size_t n_samples_volume = 200'000;

    mc::geom::Point<2> x0{};

    mc::integrators::MontecarloIntegrator<2> integrator(domain);
    mc::integrators::MHMontecarloIntegrator<2> mhintegrator(domain);
    mc::proposals::UniformProposal<2> dummy_proposal(domain);

    mhintegrator.setConfig(
        burn_in,
        thinning,
        n_samples_volume,
        deviation,
        indicator,
        x0
    );

    auto run_case = [&](const std::string& name,
                        const std::function<double(const mc::geom::Point<2>&)>& f,
                        double exact_or_nan)
    {
        const unsigned int seed = mc::rng::get_global_seed();

        const double mh_est = mhintegrator.integrate(
            f, static_cast<int>(n_samples), dummy_proposal, seed);

        const double mc_est = integrator.OLDintegrate(f, n_samples);

        auto print_line = [&](const std::string& method, double est) {
            std::cout << "  " << method << ": " << est;
            if (!std::isnan(exact_or_nan)) {
                const double abs_err = std::abs(est - exact_or_nan);
                const double rel_err = abs_err / (std::abs(exact_or_nan) + 1e-30);
                std::cout << " | abs_err=" << abs_err << " | rel_err=" << rel_err;
            }
            std::cout << "\n";
        };

        std::cout << "\n===========================================\n";
        std::cout << "Case: " << name << "\n";
        if (!std::isnan(exact_or_nan))
            std::cout << "Exact: " << exact_or_nan << "\n";
        else
            std::cout << "Exact: (no closed form used here)\n";

        print_line("MH", mh_est);
        print_line("MC", mc_est);
    };

    std::cout << "Running Benchmarks (MC vs MH) on disk R=" << R << "\n";
    std::cout << "Config: n_samples=" << n_samples
              << " burn_in=" << burn_in
              << " thinning=" << thinning
              << " deviation=" << deviation
              << " n_samples_volume=" << n_samples_volume
              << "\n";

    // --------------------------
    // A) Sanity check: r^2
    // f(x)=x^2+y^2, exact = 5000*pi for R=10
    // --------------------------
    {
        auto fA = [](const mc::geom::Point<2>& x) -> double {
            return x[0]*x[0] + x[1]*x[1];
        };
        const double exactA = 0.5 * M_PI * std::pow(R, 4); // π R^4 / 2
        run_case("A) f=r^2 (sanity check)", fA, exactA);
    }

    // --------------------------
    // B) Volume test: f(x)=1
    // exact = area = pi*R^2
    // --------------------------
    {
        auto fB = [](const mc::geom::Point<2>&) -> double { return 1.0; };
        const double exactB = M_PI * R * R;
        run_case("B) f=1 (volume)", fB, exactB);
    }

    // --------------------------
    // C) Centered Gaussian: exp(-alpha r^2)
    // exact = (pi/alpha)*(1 - exp(-alpha R^2))
    // --------------------------
    {
        const double alpha = 1.0;
        auto fC = [alpha](const mc::geom::Point<2>& x) -> double {
            const double r2 = x[0]*x[0] + x[1]*x[1];
            return std::exp(-alpha * r2);
        };
        const double exactC = (M_PI/alpha) * (1.0 - std::exp(-alpha * R * R));
        run_case("C) f=exp(-alpha r^2), alpha=1", fC, exactC);
    }

    // --------------------------
    // D) Thin ring: exp(-beta (r-r0)^2)
    // (closed form on finite disk is messy -> skip exact)
    // --------------------------
    {
        const double r0 = 8.0;
        const double beta = 50.0;
        auto fD = [r0,beta](const mc::geom::Point<2>& x) -> double {
            const double r = std::sqrt(x[0]*x[0] + x[1]*x[1]);
            const double d = r - r0;
            return std::exp(-beta * d * d);
        };
        run_case("D) f=exp(-beta (r-r0)^2) (thin ring)", fD,
                 std::numeric_limits<double>::quiet_NaN());
    }

    // --------------------------
    // E) Boundary-stressing function: 1/sqrt((R-r)+eps)
    // (integrable; exact exists but not worth here -> skip exact)
    // --------------------------
    {
        const double eps = 1e-3;
        auto fE = [eps](const mc::geom::Point<2>& x) -> double {
            const double r = std::sqrt(x[0]*x[0] + x[1]*x[1]);
            return 1.0 / std::sqrt((R - r) + eps);
        };
        run_case("E) f=1/sqrt((R-r)+eps) (boundary layer)", fE,
                 std::numeric_limits<double>::quiet_NaN());
    }
}

void runBenchmarks(const std::string& expression, bool useGnuplot) {
    n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 16;

    circleIntegrationParser(expression, useGnuplot);
    sphereIntegrationParser(expression, useGnuplot);
    rectangularIntegrationParser(expression, useGnuplot);
    cylinderIntegrationParser(expression, useGnuplot);
}
