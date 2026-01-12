//
// Created by Giacomo Merlo on 12/01/26.
//

#ifndef MONTECARLO_DGL_ISMEANESTIMATOR_HPP
#define MONTECARLO_DGL_ISMEANESTIMATOR_HPP

template <std::size_t dim>
struct ImportanceEstimate {
    double mean   = 0.0;     // estimate of E_q[h(X)] (or integral, depending on h)
    double stderr = 0.0;     // standard error (i.i.d. assumption)
    std::size_t n_samples = 0;
    std::size_t n_inside = 0;

};

template <std::size_t dim>
class ISMeanEstimator {
public:
    ImportanceEstimate<dim> estimate(const IntegrationDomain<dim>& domain,
             std::uint32_t seed,
             std::size_t n_samples,
             const Proposal<dim>& proposal,
             const function<double(const Point<dim>&)>& f) const;
};

#include "ISMeanEstimator.tpp"

#endif //MONTECARLO_DGL_ISMEANESTIMATOR_HPP