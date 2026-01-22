/**
 * @file MCintegrator.tpp
 * @brief MontecarloIntegrator template implementation.
 * @details Contains inline implementations for classic uniform Monte Carlo
 * and importance-weighted integration.
 */

// MCintegrator.tpp
#include "integrator.hpp"
#include "../geometry.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace mc::integrators {

/**
 * @brief Construct a uniform Monte Carlo integrator.
 * @tparam dim Dimensionality parameter.
 * @param d Reference to the integration domain.
 */
template <size_t dim>
MontecarloIntegrator<dim>::MontecarloIntegrator(const mc::domains::IntegrationDomain<dim>& d)
    : Integrator<dim>(d) {}

/**
 * @brief Legacy integration using pre-sampled points.
 * @tparam dim Dimensionality parameter.
 * @param f Integrand function.
 * @param n_samples Number of sample points.
 * @return Estimated integral value.
 * 
 * @deprecated Use integrate() instead.
 * @details Uses initializeRandomizer() to generate uniform samples,
 * evaluates f at inside points, and normalizes by volume.
 */
template <size_t dim>
double MontecarloIntegrator<dim>::OLDintegrate(const std::function<double(const mc::geom::Point<dim>&)>& f,
                                               int n_samples)
{
    std::vector<mc::geom::Point<dim>> points = this->initializeRandomizer(n_samples);

    double sum = 0.0;
    for (const auto& p : points) {
        if (this->domain.isInside(p)) sum += f(p);
    }

    const double volume = this->domain.getBoxVolume();
    return (sum / static_cast<double>(n_samples)) * volume;
}

/**
 * @brief Compute the integral using uniform Monte Carlo sampling.
 * @tparam dim Dimensionality parameter.
 * @param f Integrand function: ℝⁿ → ℝ.
 * @param n_samples Number of sample points to evaluate.
 * @param proposal Ignored (for API consistency with importance sampling).
 * @param seed Random seed for reproducibility.
 * @return Estimated value of ∫_Ω f(x) dx.
 * 
 * @details Algorithm:
 * 1. Uses MCMeanEstimator to sample uniformly from bounding box
 * 2. Evaluates f at points inside Ω (zero outside)
 * 3. Computes mean: μ̂ = (1/N) ∑ f(xᵢ)
 * 4. Returns integral: V_Ω · μ̂
 */
template <size_t dim>
double MontecarloIntegrator<dim>::integrate(const std::function<double(const mc::geom::Point<dim>&)>& f,
                                            int n_samples,
                                            const mc::proposals::Proposal<dim>&,
                                            std::uint32_t seed)
{
    mc::estimators::MCMeanEstimator<dim> mean_estimator;
    mc::estimators::MeanEstimate<dim> mean_estimate =
        mean_estimator.estimate(this->domain, seed, static_cast<std::size_t>(n_samples), f);

    return mean_estimate.mean * this->domain.getBoxVolume();
}

} // namespace mc::integrators