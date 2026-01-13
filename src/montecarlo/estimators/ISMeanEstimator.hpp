/**
 * @file ISMeanEstimator.hpp
 * @brief Importance sampling mean estimator
 * @author Giacomo Merlo
 * @date 12/01/26
 * 
 * Computes the mean of weighted samples for importance sampling integration.
 * Estimates ∫ f(x) dx = V * E_q[f(X)/q(X)] where q is the proposal and V is volume.
 */

#ifndef MONTECARLO_DGL_ISMEANESTIMATOR_HPP
#define MONTECARLO_DGL_ISMEANESTIMATOR_HPP

/**
 * @brief Result of importance sampling mean estimation
 * @tparam dim Dimensionality
 * 
 * Contains the estimated mean, standard error (assuming i.i.d. samples),
 * number of samples collected, and number of samples inside domain.
 */
template <std::size_t dim>
struct ImportanceEstimate {
    double mean   = 0.0;     ///< Estimated E_q[f(X)/q(X)]
    double stderr = 0.0;     ///< Standard error (i.i.d. assumption)
    std::size_t n_samples = 0; ///< Total samples generated
    std::size_t n_inside = 0;  ///< Samples that fell inside domain

};

/**
 * @brief Importance sampling estimator
 * @tparam dim Dimensionality
 * 
 * Computes mean of f(X)/q(X) where X ~ q (proposal distribution).
 * The integral is then V * mean, where V is the domain volume.
 */
template <std::size_t dim>
class ISMeanEstimator {
public:
    /**
     * @brief Estimate mean using importance sampling
     * @param domain Integration domain (for volume and containment checks)
     * @param seed Random seed for reproducibility
     * @param n_samples Number of samples to generate
     * @param proposal Proposal distribution q(x) to sample from
     * @param f Integrand function to weight samples
     * @return ImportanceEstimate with mean, error, and sampling statistics
     */
    ImportanceEstimate<dim> estimate(const IntegrationDomain<dim>& domain,
             std::uint32_t seed,
             std::size_t n_samples,
             const Proposal<dim>& proposal,
             const function<double(const Point<dim>&)>& f) const;
};

#include "ISMeanEstimator.tpp"

#endif //MONTECARLO_DGL_ISMEANESTIMATOR_HPP