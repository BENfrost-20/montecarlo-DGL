/**
 * @file gaussianProposal.hpp
 * @brief Diagonal multivariate Gaussian proposal over a generic integration domain.
 *
 * Draws samples from N(mu, diag(sigma^2)) and enforces x ∈ D via rejection:
 *   sample x ~ N(...)
 *   while x ∉ D: resample
 *
 * IMPORTANT NOTE ABOUT NORMALIZATION:
 * Rejection induces a truncated Gaussian on D:
 *   q_trunc(x) = phi(x) * 1_D(x) / Z,  Z = P(X ∈ D)
 * For generic domains Z is hard to compute. Therefore pdf() returns
 * the UNNORMALIZED density: phi(x) * 1_D(x).
 * This is fine for self-normalized IS/MIS where the unknown constant cancels.
 */

#ifndef MONTECARLO_1_GAUSSIAN_PROPOSAL_HPP
#define MONTECARLO_1_GAUSSIAN_PROPOSAL_HPP

#include "proposal.hpp"
#include "../domains/integration_domain.hpp"

#include <array>
#include <cmath>
#include <random>
#include <vector>

template <size_t dim>
class GaussianProposal : public Proposal<dim>
{
public:
    GaussianProposal(const IntegrationDomain<dim>& d,
                     const std::vector<double>& mean,
                     const std::vector<double>& sigma);

    geom::Point<dim> sample(std::mt19937& rng) const override;

    double pdf(const geom::Point<dim>& x) const override;

private:
    const IntegrationDomain<dim>& domain;

    std::vector<double> mu;
    std::vector<double> sig;
    std::vector<double> inv_sig2;     // 1/sigma^2 for each dimension

    double log_norm_const = 0.0;      // log((2pi)^(-d/2) * prod(1/sigma_i))

    // One normal distribution per dimension (diagonal covariance).
    mutable std::array<std::normal_distribution<double>, dim> ndist{};
};

#include "gaussianProposal.tpp"

#endif // MONTECARLO_1_GAUSSIAN_PROPOSAL_HPP