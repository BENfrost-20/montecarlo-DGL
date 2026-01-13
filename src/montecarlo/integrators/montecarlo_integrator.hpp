/**
 * @file montecarlo_integrator.hpp
 * @brief Monte Carlo integration engine with multiple sampling strategies
 * 
 * Implements classic Monte Carlo, importance sampling, and Metropolis-Hastings
 * integration methods for computing definite integrals over complex domains.
 */

#ifndef MONTECARLO_1_MONTECARLO_INTEGRATOR_HPP
#define MONTECARLO_1_MONTECARLO_INTEGRATOR_HPP

#include "../domains/integration_domain.hpp"
#include "../proposals/proposal.hpp"
#include "../mcmc/metropolisHastingsSampler.hpp"
#include "../estimators/VolumeEstimatorMC.hpp"
#include "../estimators/ISMeanEstimator.hpp"
#include "../geometry.hpp"
#include "integrator.hpp"
#include <functional>

/**
 * @brief Primary Monte Carlo integration class supporting multiple sampling strategies
 * @tparam dim Dimensionality of the integration domain
 * 
 * Provides three integration methods:
 * 1. Classic uniform Monte Carlo (integrate)
 * 2. Importance sampling with custom proposal (integrate_importance)
 * 3. Metropolis-Hastings MCMC sampling (integrate_with_mh)
 * 
 * All methods estimate ∫_D f(x) dx for a given domain D and integrand f.
 */
template <std::size_t dim>
class MontecarloIntegrator : public Integrator<dim> {
public:
    /**
     * @brief Construct integrator for a specific domain
     * @param d Integration domain (hypersphere, rectangle, cylinder, polytope, etc.)
     */
    explicit MontecarloIntegrator(const IntegrationDomain<dim> &d);

    /**
     * @brief Classic Monte Carlo integration with uniform sampling
     * @param f Integrand function to integrate
     * @param n_samples Number of Monte Carlo samples
     * @return Estimated integral value ∫_D f(x) dx
     * 
     * Uses Hit-or-Miss sampling within the bounding box. Error scales as O(1/√n).
     */
    // Calcola l'integrale di una funzione 'f' usando Monte Carlo
    double integrate(const std::function<double(const Point<dim>&)>& f, int n_samples);

    /**
     * @brief Importance sampling Monte Carlo integration
     * @param f Integrand function
     * @param n_samples Number of samples
     * @param proposal Custom sampling distribution (should approximate f)
     * @param seed Random seed for reproducibility
     * @return Estimated integral with reduced variance
     * 
     * Samples from proposal distribution q(x) instead of uniform.
     * Computes ∫ f(x)/q(x) * q(x) dx using importance weights.
     */
    double integrate_importance(const std::function<double(const Point<dim>&)>& f, int n_samples, const Proposal<dim>& proposal, uint32_t seed);

    /**
     * @brief Metropolis-Hastings MCMC integration
     * @param f Integrand function
     * @param p Target density (typically uniform over domain)
     * @param x0 Initial point for MCMC chain
     * @param deviation Step size for random walk proposal
     * @param seed Random seed
     * @param burn_in Number of samples to discard before collecting
     * @param n_samples Number of samples to collect (after burn-in)
     * @param thinning Keep every k-th sample to reduce autocorrelation
     * @param n_samples_volume Samples for initial volume estimation
     * @return Estimated integral = volume * mean(f)
     * 
     * Uses MH sampling to explore the domain more efficiently than uniform sampling.
     * Combines volume estimation (Hit-or-Miss) with MCMC mean estimation.
     */
    double integrate_with_mh(const std::function<double(const geom::Point<dim>&)>& f,
                                 const std::function<double(const geom::Point<dim>&)>& p,
                                 geom::Point<dim> x0,
                                 double deviation,
                                 uint32_t seed,
                                 std::size_t burn_in,
                                 std::size_t n_samples,
                                 std::size_t thinning,
                                 std::size_t n_samples_volume);
};

#include "montecarlo_integrator.tpp"

#endif // MONTECARLO_1_MONTECARLO_INTEGRATOR_HPP
