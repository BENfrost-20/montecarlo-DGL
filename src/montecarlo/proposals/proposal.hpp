/**
 * @file proposal.hpp
 * @brief Abstract base for probability distributions used in importance sampling
 * 
 * Defines the interface for proposal distributions q(x) used in importance sampling
 * and MCMC methods. Concrete implementations provide efficient sampling and PDF evaluation.
 */

#ifndef MONTECARLO_1_PROPOSAL_HPP
#define MONTECARLO_1_PROPOSAL_HPP

#include <random>
#include "../geometry.hpp"

/**
 * @brief Abstract proposal distribution interface
 * @tparam dim Dimensionality of the distribution
 * 
 * A proposal distribution q(x) is used in importance sampling to approximate
 * the integrand or in MCMC to propose candidate moves. The efficiency of
 * these algorithms depends on how well q(x) approximates the target.
 */
template <size_t dim>
class Proposal
{
public:
    /// Virtual destructor for proper cleanup
    virtual ~Proposal() = default;

    /**
     * @brief Draw a random sample from the proposal distribution
     * @param rng Mersenne Twister random generator
     * @return Point distributed according to q(x)
     */
    // Sample a point according to p(x)
    virtual geom::Point<dim> sample(std::mt19937& rng) const = 0;

    /**
     * @brief Evaluate the proposal probability density function
     * @param x Query point
     * @return q(x), the probability density at x
     * 
     * Must be non-zero everywhere in the support of the integrand.
     */
    // Evaluate p(x) (the PDF) at x
    virtual double pdf(const geom::Point<dim>& x) const = 0;
};

#endif