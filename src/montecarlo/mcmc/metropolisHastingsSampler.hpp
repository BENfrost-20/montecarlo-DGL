/**
 * @file metropolisHastingsSampler.hpp
 * @brief Metropolis-Hastings MCMC sampling engine.
 * @author Giacomo Merlo
 * @date 11/01/26
 * 
 * @details Implements the Metropolis-Hastings algorithm for sampling from arbitrary
 * target probability distributions using Markov Chain Monte Carlo (MCMC).
 * 
 * **Algorithm Overview**:
 * The Metropolis-Hastings sampler generates a sequence of correlated samples
 * distributed according to a target probability density π(x) by:
 * 1. Starting at an initial state x₀
 * 2. Proposing x' = x + N(0, σ²) from a symmetric random walk
 * 3. Computing acceptance ratio α = min(1, π(x')/π(x))
 * 4. Accepting x' with probability α, otherwise staying at x
 * 
 * **Advantages over rejection sampling**:
 * - Works efficiently in high dimensions where rejection becomes infeasible
 * - Handles arbitrary complex distributions via their density function
 * - Produces unbiased samples after burn-in period
 * 
 * **Key properties**:
 * - Samples form a Markov chain (current sample depends only on previous)
 * - Asymptotically distributed according to π(x)
 * - Symmetric random walk proposal ensures detailed balance
 * - Acceptance rate should be ~23.4% in high dimensions (optimal)
 * 
 * @note Requires careful burn-in and thinning for reliable estimates.
 * @see Metropolis et al. (1953), Hastings (1970)
 */

#ifndef MONTECARLO_DGL_METROPOLISHASTINGSSAMPLER_HPP
#define MONTECARLO_DGL_METROPOLISHASTINGSSAMPLER_HPP

#include "../domains/integration_domain.hpp"
#include "../geometry.hpp"

#include <array>
#include <random>
#include <utility>
#include <functional>

namespace mc{
namespace mcmc{


/**
 * @class MetropolisHastingsSampler
 * @brief Metropolis-Hastings MCMC sampler for arbitrary target distributions.
 * @tparam dim Dimensionality of the sample space.
 * 
 * @details Generates a Markov chain of samples asymptotically distributed
 * according to target_pdf(x) using the Metropolis-Hastings algorithm.
 * 
 * **Sampling process**:
 * - Proposal: x' = x + N(0, σ²) where σ = deviation
 * - Acceptance: α = min(1, π(x')/π(x))
 * - Update: x_{n+1} = x' with probability α, else x_{n+1} = x_n
 * 
 * **Usage**:
 * 1. Construct sampler with target density and initial point
 * 2. Call next() repeatedly to generate Markov chain
 * 3. Discard initial burn-in samples
 * 4. Use remaining samples for Monte Carlo integration
 * 5. Monitor acceptance_rate() to tune deviation
 * 
 * **Performance tuning**:
 * - If acceptance_rate() is too high (>50%), increase deviation
 * - If acceptance_rate() is too low (<10%), decrease deviation
 * - Target ~23.4% for high-dimensional problems
 * 
 * @note The target density should return 0 outside the domain rather than
 *       relying on explicit domain checks (for flexibility).
 * @warning Samples are correlated; use thinning for variance reduction.
 */
template <size_t dim>
class MetropolisHastingsSampler
{
public:
    /**
     * @brief Construct Metropolis-Hastings sampler with target distribution.
     * @param d Integration domain (defines valid region and constraints).
     * @param p Target probability density function π(x). Should return 0 outside
     *          the domain and positive values inside. Must be finite and > 0 at x0.
     * @param x0 Initial point for the Markov chain. Must satisfy domain.isInside(x0).
     * @param deviation Standard deviation σ of random walk proposal N(0, σ²).
     *                  Tune this parameter to achieve ~23.4% acceptance rate.
     * @throws std::invalid_argument If x0 is outside the domain.
     *
     * @details Initializes sampler state and validates initial conditions.
     *          The random walk proposal will use a normal distribution with
     *          standard deviation `deviation` for each dimension.
     */
    explicit MetropolisHastingsSampler(const mc::domains::IntegrationDomain<dim>& d,
                                       const std::function<double(const mc::geom::Point<dim>&)>& p,
                                       mc::geom::Point<dim> x0,
                                       double deviation);

    /**
     * @brief Generate the next sample from the Markov chain.
     * @param rng Random number generator (must be seeded externally).
     * @return Next point in the chain: either the proposed point or current point.
     *
     * @details Implements one iteration of the Metropolis-Hastings algorithm:
     * 1. Propose y = current + N(0, σ²) where σ = deviation
     * 2. Compute target densities: π(current), π(y)
     * 3. Compute acceptance ratio: α = min(1, π(y)/π(current))
     * 4. Accept y with probability α, update current = y and increment counter
     * 5. Return updated current point
     *
     * **Important**: The target density p(x) is expected to handle domain
     * constraints by returning 0 outside valid regions (not throwing exceptions).
     *
     * @throws std::runtime_error If π(current) becomes ≤ 0 or non-finite.
     *         This indicates the algorithm entered an invalid state.
     *
     * @note Automatically tracks acceptance statistics via n_steps and n_accept.
     * @note Call acceptance_rate() to monitor chain quality.
     */
    mc::geom::Point<dim> next(std::mt19937& rng);

    /**
     * @brief Evaluate target probability density at a point.
     * @param x Query point in the domain.
     * @return π(x), the target probability density at x.
     *
     * @details Evaluates the target probability density function provided
     * at construction. This is a simple accessor for the internal target.
     *
     * @note For points outside the domain, the target should return 0.
     */
    double target_pdf(const mc::geom::Point<dim>& x);

    /**
     * @brief Get the current acceptance rate of the Markov chain.
     * @return Fraction of proposed moves that were accepted: n_accept / n_steps.
     *         Returns 0.0 if no steps have been taken.
     *
     * @details The acceptance rate indicates chain quality:
     * - **Too high (>50%)**: Increase deviation to explore wider
     * - **Optimal (~23.4%)**: Good for high-dimensional problems (d > 5)
     * - **Too low (<10%)**: Decrease deviation to accept more proposals
     *
     * Use this metric to tune the deviation parameter for efficiency.
     * Generally aim for 20-40% in practice depending on dimensionality.
     *
     * @note This is a cumulative rate across all next() calls.
     *       Reset the sampler to start fresh statistics if needed.
     */
    double acceptance_rate() const;

private:
    const mc::domains::IntegrationDomain<dim>& domain;

    std::function<double(const mc::geom::Point<dim>&)> target;

    mc::geom::Point<dim> current;

    std::size_t n_steps = 0;
    std::size_t n_accept = 0;

    std::normal_distribution<double> rw_normal;
    std::uniform_real_distribution<double> uni;
};

} //namespace mcmc
} //namespace mc

#include "metropolisHastingsSampler.tpp"

#endif //MONTECARLO_DGL_METROPOLISHASTINGSSAMPLER_HPP