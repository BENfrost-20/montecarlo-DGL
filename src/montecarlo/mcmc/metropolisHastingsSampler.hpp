/**
 * @file metropolisHastingsSampler.hpp
 * @brief Metropolis-Hastings MCMC sampling engine
 * @author Giacomo Merlo
 * @date 11/01/26
 * 
 * Implements the Metropolis-Hastings algorithm for sampling from arbitrary
 * target distributions. Used to generate samples more efficiently
 * than uniform rejection sampling over complex domains.
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
 * @brief Metropolis-Hastings MCMC sampler
 * @tparam dim Dimensionality of the sample space
 * 
 * Generates a Markov chain of samples distributed according to target_pdf(x).
 * Uses a symmetric random walk proposal: x' = x + N(0, deviation²).
 * Samples are accepted with probability min(1, π(x')/π(x)).
 * 
 * Useful for sampling from high-dimensional or complex distributions
 * where direct sampling is infeasible.
 */
template <size_t dim>
class MetropolisHastingsSampler
{
public:
    /**
     * @brief Construct MH sampler with target distribution
     * @param d Integration domain (defines valid region)
     * @param p Target probability density function
     * @param x0 Initial point for Markov chain
     * @param deviation Standard deviation of random walk proposal
     */
    explicit MetropolisHastingsSampler(const mc::domains::IntegrationDomain<dim>& d,
                                       const std::function<double(const mc::geom::Point<dim>&)>& p,
                                       mc::geom::Point<dim> x0,
                                       double deviation);

    /**
     * @brief Generate next sample from Markov chain
     * @param rng Random number generator
     * @return Next point in the chain
     * 
     * Proposes x' = current + N(0, σ²) and accepts with probability
     * min(1, p(x')/p(current)). Updates acceptance statistics.
     */
    mc::geom::Point<dim> next(std::mt19937& rng);

    /**
     * @brief Evaluate target probability at point
     * @param x Query point
     * @return π(x)
     */
    double target_pdf(const mc::geom::Point<dim>& x);

    /**
     * @brief Get current acceptance rate
     * @return Fraction of proposed moves that were accepted
     * 
     * Ideal acceptance rate for high dimensions is ~0.234.
     * Adjust proposal deviation if rate is too high/low.
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