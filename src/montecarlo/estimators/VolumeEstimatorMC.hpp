/**
 * @file VolumeEstimatorMC.hpp
 * @brief Hit-or-Miss Monte Carlo volume estimation
 * @author Giacomo Merlo
 * @date 12/01/26
 * 
 * Estimates the volume of complex domains using acceptance-rejection sampling:
 * V ≈ V_box * (# hits) / (# total samples)
 */

#ifndef MONTECARLO_DGL_VOLUMEESTIMATORMC_HPP
#define MONTECARLO_DGL_VOLUMEESTIMATORMC_HPP

#include <random>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include "../domains/integration_domain.hpp"
#include "../geometry.hpp"

/**
 * @brief Result of volume estimation
 * @tparam dim Dimensionality
 * 
 * Contains estimated volume, standard error, inside ratio (points in domain),
 * and total samples generated.
 */
template <std::size_t dim>
struct VolumeEstimate {
    double volume = 0.0;     ///< Estimated |D| (domain volume)
    double stderr = 0.0;     ///< Standard error estimate
    double inside_ratio = 0.0; ///< Fraction of samples inside (p-hat)
    std::size_t n_samples = 0; ///< Total samples used
};

/**
 * @brief Hit-or-Miss volume estimator
 * @tparam dim Dimensionality
 * 
 * Estimates domain volume by sampling uniformly in bounding box and counting
 * how many points fall inside. The ratio (hits/total) * (box_volume) estimates
 * the domain volume. Standard error decreases as O(1/√n).
 */
template <std::size_t dim>
class VolumeEstimatorMC {
public:
    /**
     * @brief Estimate domain volume via Hit-or-Miss
     * @param domain Integration domain to measure
     * @param seed Random seed for reproducibility
     * @param n_samples Number of samples to generate
     * @return VolumeEstimate with estimated volume and error
     */
    VolumeEstimate<dim>
    estimate(const IntegrationDomain<dim>& domain,
             uint32_t seed,
             std::size_t n_samples) const;
};

#include "VolumeEstimatorMC.tpp"

#endif //MONTECARLO_DGL_VOLUMEESTIMATORMC_HPP