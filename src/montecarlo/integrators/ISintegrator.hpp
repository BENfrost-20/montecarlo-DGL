//
// Created by Giacomo Merlo on 15/01/26.
//

#ifndef MONTECARLO_DGL_ISINTEGRATOR_HPP
#define MONTECARLO_DGL_ISINTEGRATOR_HPP

#include "../domains/integration_domain.hpp"
#include "../proposals/proposal.hpp"
#include "../proposals/uniformProposal.hpp"
#include "../proposals/gaussianProposal.hpp"
#include "../proposals/mixtureProposal.hpp"
#include "../mcmc/metropolisHastingsSampler.hpp"
#include "../estimators/VolumeEstimatorMC.hpp"
#include "../estimators/ISMeanEstimator.hpp"
#include "../estimators/MCMeanEstimator.hpp"
#include "../geometry.hpp"
#include "integrator.hpp"
#include <functional>

namespace mc::integrators {

template <std::size_t dim>
class ISMontecarloIntegrator : public Integrator<dim> {
public:

    explicit ISMontecarloIntegrator(const mc::domains::IntegrationDomain<dim> &d);

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
    double integrate(const std::function<double(const mc::geom::Point<dim>&)>& f,
                     int n_samples,
                     const mc::proposals::Proposal<dim>& proposal,
                     std::uint32_t seed) override;

};

} // namespace mc::integrators

#include "ISintegrator.tpp"

#endif //MONTECARLO_DGL_ISINTEGRATOR_HPP