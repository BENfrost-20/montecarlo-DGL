// gaussianProposal.hpp
/**
 * @file gaussianProposal.hpp
 * @brief Diagonal multivariate Gaussian proposal (no domain truncation).
 *
 * IMPORTANT:
 * - sample(rng) draws from the full Gaussian in R^dim (NO rejection).
 * - pdf(x) returns the corresponding full Gaussian density (NO domain indicator).
 *
 * Domain constraints (if any) must be handled by the estimator via:
 *   if(domain.isInside(p)) { ... }
 *
 * This guarantees sample() and pdf() are always coherent with the Proposal interface.
 */

#ifndef MONTECARLO_1_GAUSSIAN_PROPOSAL_HPP
#define MONTECARLO_1_GAUSSIAN_PROPOSAL_HPP

#include "proposal.hpp"
#include "../domains/integration_domain.hpp"

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace mc {
namespace proposals {

template <size_t dim>
class GaussianProposal : public Proposal<dim>
{
public:
    GaussianProposal(const mc::domains::IntegrationDomain<dim>& d,
                     const std::vector<double>& mean,
                     const std::vector<double>& sigma);

    mc::geom::Point<dim> sample(std::mt19937& rng) const override;
    double pdf(const mc::geom::Point<dim>& x) const override;

private:
    const mc::domains::IntegrationDomain<dim>& domain; // kept for consistency / future use

    std::vector<double> mu;
    std::vector<double> sig;
    std::vector<double> inv_sig2; // 1/(sigma^2)

    double log_norm_const = 0.0; // log((2pi)^(-d/2) * prod_i (1/sigma_i))

    void init_from_mu_sig_();
};

} // namespace proposals
} // namespace mc

#include "gaussianProposal.tpp"

#endif // MONTECARLO_1_GAUSSIAN_PROPOSAL_HPP