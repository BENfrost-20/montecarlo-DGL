/**
 * @file metropolisHastingsSampler.tpp
 * @brief MetropolisHastingsSampler template implementation.
 * @details Implements the Metropolis-Hastings MCMC algorithm with
 * symmetric random walk proposal and detailed balance verification.
 */

//
// Created by Giacomo Merlo on 11/01/26.
//


#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <stdexcept>

namespace mc{
namespace mcmc{

/**
 * @brief Construct and validate the Metropolis-Hastings sampler.
 * @tparam dim Template dimension parameter.
 * @param d Integration domain constraint.
 * @param p Target probability density function.
 * @param x0 Initial state of the Markov chain.
 * @param deviation Standard deviation of random walk proposal.
 * 
 * @throws std::invalid_argument If initial point x0 is outside the domain.
 * 
 * @details Initializes the sampler with:
 * - Reference to the domain constraint (used in next() for validation)
 * - Target density function p (called at each proposal)
 * - Current state = x0 (starting point of chain)
 * - Random walk parameters: N(0, deviation²) for each coordinate
 * - Acceptance statistics: n_steps = 0, n_accept = 0
 * 
 * The constructor validates that x0 is within the domain; otherwise
 * the Markov chain cannot start in a valid state.
 */
template <size_t dim>
MetropolisHastingsSampler<dim>::MetropolisHastingsSampler(
    const mc::domains::IntegrationDomain<dim>& d,
    const std::function<double(const mc::geom::Point<dim>&)>& p,
    mc::geom::Point<dim> x0,
    double deviation)
  : domain(d)
  , target(p)
  , current(x0)
  , rw_normal(0.0, deviation)
  , uni(0.0, 1.0)
{
    if (!domain.isInside(current)) {
        throw std::invalid_argument("MetropolisHastingsSampler: x0 is outside the domain.");
    }
}

/**
 * @brief Generate the next sample in the Markov chain.
 * @tparam dim Template dimension parameter.
 * @param rng Random number generator engine.
 * @return The current state after update: either y (accepted) or x (rejected).
 * 
 * @throws std::runtime_error If target(current) becomes ≤ 0 or non-finite.
 *         This indicates an algorithm failure.
 * 
 * @details Implements one step of Metropolis-Hastings:
 * 
 * **Step 1: Proposal**
 * - Generate y = current + ε where ε ~ N(0, σ²) for each coordinate
 * - σ is the deviation parameter specified at construction
 * 
 * **Step 2: Domain & Validity Check**
 * - Evaluate π(y). If non-finite or ≤ 0, reject automatically
 * - (Relies on target density returning 0 outside domain, not domain checks)
 * 
 * **Step 3: Acceptance Ratio**
 * - Evaluate π(current). Must be finite and > 0 (invariant)
 * - Compute α = min(1, π(y)/π(current))
 * 
 * **Step 4: Accept/Reject**
 * - Draw u ~ Unif[0,1]
 * - If u < α: accept y (current = y, n_accept++)
 * - Else: reject y (current unchanged)
 * 
 * **Step 5: Statistics**
 * - Increment n_steps
 * - Return updated current
 * 
 * @note Symmetric proposal (N(0,σ²)) ensures detailed balance holds.\n
 *       The ratio simplifies to π(y)/π(current) without proposal ratio.\n
 *       Domain handling via target density return value adds flexibility.
 * 
 * @see acceptance_rate() for monitoring chain behavior
 */
template <size_t dim>
geom::Point<dim>
MetropolisHastingsSampler<dim>::next(std::mt19937& rng)
{
    ++n_steps;

    mc::geom::Point<dim> y = current;
    for (std::size_t k = 0; k < dim; ++k)
        y[k] += rw_normal(rng);

    // NOTE: Domain handling is delegated to the target density function.
    // The target should return 0 outside the valid domain rather than
    // throwing exceptions. This approach offers greater flexibility compared
    // to explicit domain checking within the sampler.

    const double px = target(current);
    const double py = target(y);

    if (!std::isfinite(py) || py <= 0.0) {
        return current; // reject
    }

    double alpha = 0.0;

    if (!std::isfinite(px) || px <= 0.0) {
        throw std::runtime_error("MH: target(current) must be finite and > 0.");
    }else{
        alpha = std::min(1.0, py / px);
    }

    if (uni(rng) < alpha) {
        current = y;
        ++n_accept;
    }
    return current;
}

/**
 * @brief Evaluate the target probability density at a point.
 * @tparam dim Template dimension parameter.
 * @param x The query point.
 * @return π(x), the target density value at x.
 * 
 * @details Simple accessor that delegates to the target density function
 * stored at construction. This allows external code to evaluate π(x)
 * without direct access to the stored function object.
 */
template <size_t dim>
double
MetropolisHastingsSampler<dim>::target_pdf(const mc::geom::Point<dim>& x)
{
    return target(x);
}

/**
 * @brief Get the current acceptance rate of the Markov chain.
 * @tparam dim Template dimension parameter.
 * @return Acceptance rate: n_accept / n_steps, or 0.0 if n_steps == 0.
 * 
 * @details Returns the fraction of proposed moves (calls to next()) that were
 * accepted by the Metropolis-Hastings criterion. This cumulative metric helps
 * tune the proposal deviation for better exploration.
 * 
 * **Interpretation**:
 * - Acceptance rate = n_accept / n_steps
 * - Theoretically optimal ~23.4% for high-dimensional problems (d > 5)
 * - In practice, 20-40% is often reasonable depending on dimension and target
 * 
 * **Tuning advice**:
 * - If rate > 50%: increase deviation to propose larger moves
 * - If rate < 10%: decrease deviation to propose smaller, safer moves
 * - If rate ≈ 23.4%: well-tuned for high dimensions
 */
template <size_t dim>
double
MetropolisHastingsSampler<dim>::acceptance_rate() const
{
    return (n_steps == 0) ? 0.0 : static_cast<double>(n_accept) / static_cast<double>(n_steps);
}

} //namespace mcmc
} //namespace mc