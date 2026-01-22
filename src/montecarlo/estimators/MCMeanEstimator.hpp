// MCMeanEstimator.hpp
//
// Created by Giacomo Merlo on 14/01/26.
//

#ifndef MONTECARLO_DGL_MCMEANESTIMATOR_HPP
#define MONTECARLO_DGL_MCMEANESTIMATOR_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

namespace mc::estimators {

template <std::size_t dim>
struct MeanEstimate {
    double mean   = 0.0;        ///< Estimated E_q[f(X)]
    double stderr = 0.0;        ///< Standard error (i.i.d. assumption)
    std::size_t n_samples = 0;  ///< Total samples generated
    std::size_t n_inside  = 0;  ///< Samples that fell inside domain
};

template <std::size_t dim>
class MCMeanEstimator {
public:
    MeanEstimate<dim> estimate(const mc::domains::IntegrationDomain<dim>& domain,
                              std::uint32_t seed,
                              std::size_t n_samples,
                              const std::function<double(const mc::geom::Point<dim>&)>& f) const;

private:
    mutable std::array<std::uniform_real_distribution<double>, dim> dist{};
};

} // namespace mc::estimators

#include "MCMeanEstimator.tpp"

#endif // MONTECARLO_DGL_MCMEANESTIMATOR_HPP