/**
 * @file ISintegrator.tpp
 * @brief ISMontecarloIntegrator template implementation.
 * @details Contains inline implementations for importance sampling integration
 * with custom proposal distributions.
 */
#include "integrator.hpp"
#include "../geometry.hpp"
#include <vector>
#include <fstream>
#include <iostream>
#include <omp.h>

namespace mc::integrators {

/**
 * @brief Construct an importance sampling integrator.
 * @tparam dim Dimensionality parameter.
 * @param d Reference to the integration domain.
 */
template <size_t dim>
ISMontecarloIntegrator<dim>::ISMontecarloIntegrator(const mc::domains::IntegrationDomain<dim> &d)
    : Integrator<dim>(d) {}

/**
 * @brief Compute the integral using importance sampling.
 * @tparam dim Dimensionality parameter.
 * @param f Integrand function: ℝⁿ → ℝ.
 * @param n_samples Number of samples drawn from the proposal.
 * @param proposal Custom sampling distribution q(x) (should approximate f for efficiency).
 * @param seed Random seed for reproducibility.
 * @return Estimated integral ∫_Ω f(x) dx.
 * 
 * @details Algorithm:
 * 1. Uses ISMeanEstimator to sample from proposal q(x)
 * 2. Computes weighted mean: μ̂ = (1/N) ∑ [f(xᵢ)/q(xᵢ)]
 * 3. Returns mean directly (no volume factor for importance sampling)
 * 
 * **Variance reduction**: When q(x) ≈ f(x), the weights f/q are nearly constant,
 * reducing variance compared to uniform sampling.
 */
template <size_t dim>
double ISMontecarloIntegrator<dim>::integrate(
    const std::function<double(const mc::geom::Point<dim>&)>& f,
    int n_samples,
    const mc::proposals::Proposal<dim>& proposal,
    std::uint32_t seed)
{
    mc::estimators::ISMeanEstimator<dim> mean_estimator;
    mc::estimators::ImportanceEstimate<dim> mean_estimate = mean_estimator.estimate(this->domain, seed, n_samples, proposal, f);
    return mean_estimate.mean;
}

} // namespace mc::integrators
