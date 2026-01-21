/**
 * @file exotic_options_pricing.cpp
 * @brief Educational example of exotic option pricing using Monte Carlo integration
 *        combined with PSO and GA optimizers
 * 
 * @details This application demonstrates practical use of the Monte Carlo library
 * for financial derivatives pricing. It showcases:
 * 1. Monte Carlo integration for option pricing (computing expected payoffs)
 * 2. PSO (Particle Swarm Optimization) for finding optimal hedge ratios
 * 3. GA (Genetic Algorithm) for calibrating option parameters
 * 
 * **FINANCIAL CONCEPTS EXPLAINED:**
 * 
 * - **Option**: A contract giving the right (not obligation) to buy/sell an asset
 *   at a specific price (strike) on/before a specific date (maturity).
 *   Think of it like buying insurance for a stock price.
 * 
 * - **Call Option**: Right to BUY an asset at strike price K.
 *   You profit if the stock price goes UP above K.
 *   Payoff = max(S - K, 0) where S is final stock price.
 * 
 * - **Exotic Option**: More complex than standard options. Here we price:
 *   * Asian Option: Payoff depends on the AVERAGE price over time, not just final price.
 *     This is fairer and harder to manipulate than regular options.
 *   * Barrier Option: Only pays if price crosses a certain level (barrier).
 *     Like a trigger that activates the option.
 * 
 * - **Monte Carlo Pricing**: We simulate thousands of possible future stock price paths.
 *   Each path is a random walk following realistic market dynamics.
 *   Option price = average of all payoffs, discounted to today's value.
 * 
 * - **Hedging**: Reducing risk by taking offsetting positions.
 *   Like buying both an umbrella and sunglasses - one protects you either way.
 * 
 * - **Delta Hedging**: Holding a specific amount of stock to offset option risk.
 *   Delta is how much the option price changes when stock price changes by $1.
 * 
 * **WHAT THIS CODE DOES:**
 * 
 * 1. Prices an Asian option using Monte Carlo simulation
 * 2. Uses PSO to find the best hedge ratio (how much stock to hold for risk reduction)
 * 3. Uses GA to calibrate volatility parameter from market data
 * 
 * @author Generated for educational purposes
 * @date 2026
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <iomanip>
#include <random>
#include <functional>
#include <algorithm>

// Monte Carlo library components
#include "../montecarlo/geometry.hpp"
#include "../montecarlo/rng/rng_global.hpp"
#include "../montecarlo/integrators/MCintegrator.hpp"
#include "../montecarlo/domains/hyperrectangle.hpp"
#include "../montecarlo/proposals/uniformProposal.hpp"

// Optimization algorithms
#include "../montecarlo/optimizers/PSO.hpp"
#include "../montecarlo/optimizers/GA.hpp"

using namespace geom;
using namespace optimizers;

// =============================================================================
// GLOBAL PARAMETERS (Financial Market Model)
// =============================================================================

/// Stock price today (initial value)
constexpr double S0 = 100.0;

/// Strike price - the price at which we can buy the stock through the option
/// Economic meaning: If final price > K, we make money by exercising
constexpr double K = 100.0;

/// Risk-free interest rate (annualized) - like a safe government bond rate
/// Economic meaning: Money today is worth more than money tomorrow due to interest
constexpr double r = 0.05;

/// Time to maturity in years (3 months = 0.25 years)
/// Economic meaning: How long until the option expires
constexpr double T = 0.25;

/// Volatility (annualized standard deviation of returns)
/// Economic meaning: How much the stock price jumps around
/// Higher volatility = more uncertainty = more valuable options
constexpr double sigma = 0.20;

/// Number of time steps in the simulation (daily monitoring for 3 months)
constexpr int N_STEPS = 63;

/// Time increment (one day in fraction of a year)
constexpr double dt = T / N_STEPS;

/// Barrier level for barrier options (knock-in level)
/// Economic meaning: Option only activates if price crosses this barrier
constexpr double BARRIER = 110.0;

// =============================================================================
// STOCK PRICE SIMULATION ENGINE
// =============================================================================

/**
 * @brief Simulates one possible future path of a stock price
 * 
 * Uses Geometric Brownian Motion (GBM), the standard model in finance:
 * dS = μ*S*dt + σ*S*dW
 * 
 * Economic intuition:
 * - Stock grows on average at rate r (risk-neutral pricing)
 * - Random shocks (dW) add volatility proportional to sigma
 * - The path is exponential (prices can't go negative)
 * 
 * @param rng Random number generator
 * @return Vector of prices at each time step [S(0), S(dt), S(2dt), ..., S(T)]
 */
std::vector<double> simulate_stock_path(std::mt19937& rng) {
    std::vector<double> path(N_STEPS + 1);
    path[0] = S0;
    
    // Normal distribution for random shocks (market uncertainty)
    std::normal_distribution<double> normal(0.0, 1.0);
    
    // Simulate price evolution day by day
    for (int i = 1; i <= N_STEPS; ++i) {
        // Random shock: dW ~ Normal(0, sqrt(dt))
        double dW = normal(rng) * std::sqrt(dt);
        
        // Geometric Brownian Motion formula:
        // S(t+dt) = S(t) * exp((r - 0.5*σ²)*dt + σ*dW)
        // The -0.5*σ² term is a correction (Itô's lemma)
        double drift = (r - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * dW;
        path[i] = path[i-1] * std::exp(drift + diffusion);
    }
    
    return path;
}

// =============================================================================
// OPTION PAYOFF FUNCTIONS
// =============================================================================

/**
 * @brief Asian call option payoff - depends on AVERAGE price
 * 
 * Economic meaning: Instead of looking at just the final price,
 * we average all prices over the option's lifetime.
 * 
 * Why it's useful:
 * - Fairer for employee stock options (can't be manipulated at expiry)
 * - Cheaper than regular options (averaging reduces volatility)
 * - Used for commodity pricing (oil, gold) based on average costs
 * 
 * @param path Simulated stock price path
 * @return Payoff: max(average_price - strike, 0)
 */
double asian_call_payoff(const std::vector<double>& path) {
    // Compute arithmetic average of all prices in the path
    double sum = 0.0;
    for (double price : path) {
        sum += price;
    }
    double avg_price = sum / path.size();
    
    // Payoff: profit if average > strike, zero otherwise
    return std::max(avg_price - K, 0.0);
}

/**
 * @brief Up-and-In Barrier call option payoff
 * 
 * Economic meaning: This option ONLY pays off if the stock price
 * crosses above the barrier at some point. It "knocks in" when triggered.
 * 
 * Why it's useful:
 * - Cheaper than regular options (might never activate)
 * - Bets on momentum: "I think it will break through $110"
 * - Popular in structured products and corporate finance
 * 
 * @param path Simulated stock price path
 * @return Payoff if barrier was hit, 0 otherwise
 */
double barrier_call_payoff(const std::vector<double>& path) {
    // Check if price ever crossed the barrier
    bool barrier_hit = false;
    for (double price : path) {
        if (price >= BARRIER) {
            barrier_hit = true;
            break;
        }
    }
    
    // Only pay if barrier was hit
    if (barrier_hit) {
        double final_price = path.back();
        return std::max(final_price - K, 0.0);
    }
    return 0.0;
}

// =============================================================================
// MONTE CARLO PRICING ENGINE
// =============================================================================

/**
 * @brief Prices an option using Monte Carlo simulation
 * 
 * Process:
 * 1. Simulate many possible future stock price paths
 * 2. Compute the payoff for each path
 * 3. Average all payoffs (expected value)
 * 4. Discount to present value using risk-free rate
 * 
 * Economic principle: Risk-Neutral Pricing
 * - In a fair market, option price = discounted expected payoff
 * - We assume investors are risk-neutral (don't need extra return for risk)
 * 
 * @param payoff_func Function computing payoff from a price path
 * @param n_simulations Number of Monte Carlo paths (more = more accurate)
 * @param seed Random seed for reproducibility
 * @return Estimated fair price of the option today
 */
double price_option_monte_carlo(
    std::function<double(const std::vector<double>&)> payoff_func,
    int n_simulations,
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);
    
    double sum_payoffs = 0.0;
    
    // Simulate many possible futures
    for (int i = 0; i < n_simulations; ++i) {
        // Generate one possible price path
        auto path = simulate_stock_path(rng);
        
        // What would we earn on this path?
        double payoff = payoff_func(path);
        sum_payoffs += payoff;
    }
    
    // Average payoff across all scenarios
    double expected_payoff = sum_payoffs / n_simulations;
    
    // Discount to present value: PV = E[Payoff] * e^(-rT)
    // Economic meaning: $100 in 3 months is worth less than $100 today
    double discount_factor = std::exp(-r * T);
    double option_price = expected_payoff * discount_factor;
    
    return option_price;
}

// =============================================================================
// OPTIMIZATION PROBLEM 1: DELTA HEDGING WITH PSO
// =============================================================================

/**
 * @brief Finds optimal hedge ratio using Particle Swarm Optimization
 * 
 * HEDGING PROBLEM:
 * - You sold an option and need to reduce risk
 * - You hold Δ shares of stock to offset option risk
 * - Goal: Find Δ that minimizes profit/loss variance
 * 
 * Economic intuition:
 * - If stock goes up $1, option value changes by ~Δ dollars
 * - Holding -Δ shares creates offsetting position
 * - Perfect hedge: portfolio value stays constant regardless of stock moves
 * 
 * WHY PSO?
 * - Hedge ratio depends on complex Monte Carlo simulations
 * - No analytical formula (unlike simple Black-Scholes)
 * - PSO explores parameter space efficiently
 * - Particles "swarm" toward low-variance solutions
 * 
 * @param option_price Current option value
 * @param n_paths Number of scenarios to test hedge
 * @return Optimal delta (number of shares to hold)
 */
double find_optimal_delta_pso(double option_price, int n_paths = 1000) {
    
    std::cout << "\n=== DELTA HEDGING WITH PSO ===" << std::endl;
    std::cout << "Goal: Find number of shares that minimizes hedging risk\n" << std::endl;
    
    /**
     * Objective function: variance of hedged portfolio P&L
     * 
     * Portfolio = Short 1 option + Long delta shares
     * P&L = (option_payoff - option_price) - delta * (S_T - S_0)
     * 
     * We want to minimize variance of P&L across scenarios
     * Low variance = predictable outcome = good hedge
     */
    auto hedge_variance = [option_price, n_paths](const Coordinates& params) -> double {
        double delta = params[0];  // Number of shares to hold
        
        std::mt19937 rng(12345);
        std::vector<double> pnl_scenarios;
        pnl_scenarios.reserve(n_paths);
        
        // Simulate many scenarios to test this hedge ratio
        for (int i = 0; i < n_paths; ++i) {
            auto path = simulate_stock_path(rng);
            double final_stock = path.back();
            
            // Option payoff at expiry
            double option_pnl = asian_call_payoff(path) - option_price;
            
            // Stock position P&L
            double stock_pnl = delta * (final_stock - S0);
            
            // Total portfolio P&L
            double total_pnl = option_pnl - stock_pnl;
            pnl_scenarios.push_back(total_pnl);
        }
        
        // Compute variance of P&L (lower is better)
        double mean = 0.0;
        for (double pnl : pnl_scenarios) mean += pnl;
        mean /= pnl_scenarios.size();
        
        double variance = 0.0;
        for (double pnl : pnl_scenarios) {
            variance += (pnl - mean) * (pnl - mean);
        }
        variance /= pnl_scenarios.size();
        
        return variance;  // PSO will minimize this
    };
    
    // Configure PSO
    PSOConfig config;
    config.population_size = 20;
    config.max_iterations = 30;
    config.inertia_weight = 0.7;
    config.cognitive_coeff = 1.4;
    config.social_coeff = 1.4;
    
    PSO optimizer(config);
    
    // Search space: delta can range from -2.0 to 2.0 shares
    // (negative means shorting stock)
    Coordinates lower = {-2.0};
    Coordinates upper = {2.0};
    
    optimizer.setObjectiveFunction(hedge_variance);
    optimizer.setBounds(lower, upper);
    optimizer.setMode(OptimizationMode::MINIMIZE);
    
    // Run optimization
    std::cout << "Running PSO with " << config.population_size 
              << " particles for " << config.max_iterations << " iterations..." << std::endl;
    
    Solution best = optimizer.optimize();
    
    std::cout << "✓ Optimal hedge ratio (delta): " << best.params[0] << std::endl;
    std::cout << "  Hedged portfolio variance: " << best.value << std::endl;
    std::cout << "  Interpretation: Hold " << std::abs(best.params[0]) 
              << " shares " << (best.params[0] > 0 ? "(long)" : "(short)") 
              << " per option to minimize risk\n" << std::endl;
    
    return best.params[0];
}

// =============================================================================
// OPTIMIZATION PROBLEM 2: VOLATILITY CALIBRATION WITH GA
// =============================================================================

/**
 * @brief Calibrates volatility parameter using Genetic Algorithm
 * 
 * CALIBRATION PROBLEM:
 * - Market is trading the option at a known price
 * - Our model has parameter σ (volatility)
 * - Goal: Find σ that makes model price match market price
 * 
 * Economic intuition:
 * - Implied volatility: the σ value "implied" by market prices
 * - Markets often have better info than models
 * - Calibration extracts market's expectation of future volatility
 * 
 * WHY GA?
 * - Robust to noisy objective functions (Monte Carlo has randomness)
 * - Doesn't require gradients (pricing is a black box)
 * - Population-based search avoids local minima
 * - Evolution finds good solutions through crossover and mutation
 * 
 * In practice: This is called "implied volatility" calculation
 * 
 * @param market_price Target price to match
 * @param n_mc_samples Samples per pricing evaluation
 * @return Calibrated volatility parameter
 */
double calibrate_volatility_ga(double market_price, int n_mc_samples = 5000) {
    
    std::cout << "\n=== VOLATILITY CALIBRATION WITH GA ===" << std::endl;
    std::cout << "Goal: Find volatility that matches market price\n" << std::endl;
    
    /**
     * Objective: minimize squared error between model and market
     * 
     * We want: model_price(σ) ≈ market_price
     * Minimize: (model_price - market_price)²
     */
    auto pricing_error = [market_price, n_mc_samples](const Coordinates& params) -> double {
        double test_sigma = params[0];
        
        // Temporarily override global sigma for this test
        // (In production, we'd pass sigma as parameter to pricing function)
        double saved_sigma = sigma;
        const_cast<double&>(sigma) = test_sigma;
        
        // Price option with this volatility
        double model_price = price_option_monte_carlo(asian_call_payoff, n_mc_samples, 999);
        
        // Restore original sigma
        const_cast<double&>(sigma) = saved_sigma;
        
        // Return squared error
        double error = model_price - market_price;
        return error * error;
    };
    
    // Configure GA
    GAConfig config;
    config.population_size = 40;
    config.max_generations = 50;
    config.tournament_k = 3;
    config.crossover_rate = 0.8;
    config.mutation_rate = 0.2;
    config.elitism_count = 2;
    
    GA optimizer(config);
    
    // Search space: volatility typically ranges from 5% to 100%
    Coordinates lower = {0.05};
    Coordinates upper = {1.00};
    
    optimizer.setObjectiveFunction(pricing_error);
    optimizer.setBounds(lower, upper);
    optimizer.setMode(OptimizationMode::MINIMIZE);
    
    // Run optimization
    std::cout << "Running GA with population=" << config.population_size 
              << " for " << config.max_generations << " generations..." << std::endl;
    
    Solution best = optimizer.optimize();
    
    std::cout << "✓ Calibrated volatility: " << (best.params[0] * 100) << "%" << std::endl;
    std::cout << "  Pricing error: " << std::sqrt(best.value) << std::endl;
    std::cout << "  Interpretation: Market expects " << (best.params[0] * 100) 
              << "% annualized volatility\n" << std::endl;
    
    return best.params[0];
}

// =============================================================================
// MAIN DEMONSTRATION
// =============================================================================

int main() {
    // Set global seed for reproducible results
    mc::set_global_seed(20260121u);
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   EXOTIC OPTIONS PRICING WITH MONTE CARLO & OPTIMIZERS     ║\n";
    std::cout << "║   Educational Example - Non-Economists Welcome!            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    // -------------------------------------------------------------------------
    // PART 1: OPTION PRICING
    // -------------------------------------------------------------------------
    
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "PART 1: PRICING EXOTIC OPTIONS WITH MONTE CARLO\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    
    std::cout << "Market Parameters:\n";
    std::cout << "  Current stock price (S₀):    $" << S0 << std::endl;
    std::cout << "  Strike price (K):            $" << K << std::endl;
    std::cout << "  Time to maturity (T):        " << T << " years (" 
              << int(T*12) << " months)" << std::endl;
    std::cout << "  Risk-free rate (r):          " << (r*100) << "%" << std::endl;
    std::cout << "  Volatility (σ):              " << (sigma*100) << "%" << std::endl;
    std::cout << "  Barrier level:               $" << BARRIER << std::endl;
    std::cout << "\n";
    
    // Price Asian option
    std::cout << "Pricing Asian Call Option...\n";
    std::cout << "(Payoff based on average price over time)\n";
    int n_sims = 50000;
    double asian_price = price_option_monte_carlo(asian_call_payoff, n_sims);
    std::cout << "  → Fair value: $" << std::fixed << std::setprecision(4) 
              << asian_price << " (using " << n_sims << " simulations)" << std::endl;
    std::cout << "  Interpretation: If you sell this option, charge $" 
              << asian_price << " to break even on average\n" << std::endl;
    
    // Price barrier option
    std::cout << "Pricing Up-and-In Barrier Call Option...\n";
    std::cout << "(Only pays if price crosses $" << BARRIER << ")\n";
    double barrier_price = price_option_monte_carlo(barrier_call_payoff, n_sims);
    std::cout << "  → Fair value: $" << std::fixed << std::setprecision(4) 
              << barrier_price << " (using " << n_sims << " simulations)" << std::endl;
    std::cout << "  Interpretation: Cheaper than regular option (might never activate)\n" << std::endl;
    
    // -------------------------------------------------------------------------
    // PART 2: DELTA HEDGING WITH PSO
    // -------------------------------------------------------------------------
    
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "PART 2: RISK MANAGEMENT - FINDING OPTIMAL HEDGE\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    std::cout << "\nScenario: You sold an Asian call for $" << asian_price << std::endl;
    std::cout << "Problem: Stock price movements create risk (you might lose money)" << std::endl;
    std::cout << "Solution: Hold a specific number of shares to offset the risk\n" << std::endl;
    
    double optimal_delta = find_optimal_delta_pso(asian_price, 800);
    
    std::cout << "Risk Management Strategy:" << std::endl;
    std::cout << "  For every option sold, hold " << std::abs(optimal_delta) 
              << " shares of stock" << std::endl;
    std::cout << "  This minimizes your profit/loss fluctuations" << std::endl;
    std::cout << "  (Dynamic hedging: adjust this ratio as market moves)\n" << std::endl;
    
    // -------------------------------------------------------------------------
    // PART 3: PARAMETER CALIBRATION WITH GA
    // -------------------------------------------------------------------------
    
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "PART 3: MODEL CALIBRATION - EXTRACTING MARKET EXPECTATIONS\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";
    
    // Simulate market price (in reality, we'd observe this from exchanges)
    double market_price = asian_price * 1.05;  // 5% higher than our model
    
    std::cout << "\nScenario: Market is trading the option at $" << market_price << std::endl;
    std::cout << "Our model prices it at: $" << asian_price 
              << " (using σ=" << (sigma*100) << "%)" << std::endl;
    std::cout << "Question: What volatility does the market expect?\n" << std::endl;
    
    double implied_vol = calibrate_volatility_ga(market_price, 3000);
    
    std::cout << "Market Intelligence:" << std::endl;
    std::cout << "  Our model assumption:  σ = " << (sigma*100) << "%" << std::endl;
    std::cout << "  Market expectation:    σ = " << (implied_vol*100) << "%" << std::endl;
    std::cout << "  Difference:            " 
              << std::abs(implied_vol - sigma)*100 << " percentage points" << std::endl;
    std::cout << "\nThis 'implied volatility' tells us the market expects " 
              << (implied_vol > sigma ? "MORE" : "LESS") 
              << " price movement than our model\n" << std::endl;
    
    // -------------------------------------------------------------------------
    // SUMMARY
    // -------------------------------------------------------------------------
    
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "SUMMARY: WHAT WE LEARNED\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    
    std::cout << "1. PRICING (Monte Carlo Integration):\n";
    std::cout << "   • Simulated thousands of possible futures\n";
    std::cout << "   • Computed average payoff for each option type\n";
    std::cout << "   • Asian option: $" << asian_price << " (smoother payoff)\n";
    std::cout << "   • Barrier option: $" << barrier_price << " (conditional payoff)\n\n";
    
    std::cout << "2. HEDGING (PSO Optimization):\n";
    std::cout << "   • Found optimal stock position to reduce risk\n";
    std::cout << "   • Delta = " << optimal_delta << " shares per option\n";
    std::cout << "   • PSO explored hedge ratios efficiently\n";
    std::cout << "   • Minimized variance of hedged portfolio\n\n";
    
    std::cout << "3. CALIBRATION (GA Optimization):\n";
    std::cout << "   • Extracted market's volatility expectation\n";
    std::cout << "   • Implied vol: " << (implied_vol*100) << "%\n";
    std::cout << "   • GA evolved parameter through generations\n";
    std::cout << "   • Matched model to market prices\n\n";
    
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "LIBRARY COMPONENTS DEMONSTRATED:\n";
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
    std::cout << "✓ Monte Carlo Integration (simulate + average)\n";
    std::cout << "✓ PSO Optimizer (swarm-based search)\n";
    std::cout << "✓ GA Optimizer (evolutionary search)\n";
    std::cout << "✓ Stochastic processes (Geometric Brownian Motion)\n";
    std::cout << "✓ Risk-neutral pricing (financial mathematics)\n\n";
    
    std::cout << "This example shows how Monte Carlo methods solve real-world\n";
    std::cout << "financial problems where analytical solutions don't exist.\n";
    std::cout << "The optimizers help us find best parameters and strategies.\n\n";
    
    return 0;
}
