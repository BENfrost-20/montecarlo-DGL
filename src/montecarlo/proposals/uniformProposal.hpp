/**
 * @file uniformProposal.hpp
 * @brief Uniform distribution proposal over integration domain
 * 
 * Provides a uniform proposal distribution for importance sampling,
 * useful as a baseline when domain shape matches target shape.
 */

#ifndef MONTECARLO_1_UNIFORM_PROPOSAL_HPP
#define MONTECARLO_1_UNIFORM_PROPOSAL_HPP

#include "proposal.hpp"
#include "../domains/integration_domain.hpp"

#include <array>
#include <random>
#include <utility>

namespace mc {
namespace proposals {

/**
 * @brief Uniform distribution over a domain
 * @tparam dim Dimensionality
 * 
 * Proposes points uniformly distributed over the integration domain.
 * PDF is constant: q(x) = 1/V where V is the domain volume.
 * Efficient for domains where we want equal weighting across all regions.
 */
template <size_t dim>
class UniformProposal : public Proposal<dim>
{
public:
    /**
     * @brief Construct uniform proposal over domain
     * @param d Integration domain to sample from
     */
    explicit UniformProposal(const mc::domains::IntegrationDomain<dim>& d);

    /**
     * @brief Sample uniform point from domain
     * @param rng Random generator
     * @return Point uniformly distributed in domain
     */
    mc::geom::Point<dim> sample(std::mt19937& rng) const override;

    /**
     * @brief Evaluate uniform PDF
     * @param x Query point
     * @return 1/V if x in domain, undefined behavior otherwise
     */
    double pdf(const mc::geom::Point<dim>&) const override;

private:
    const mc::domains::IntegrationDomain<dim>& domain;
    mutable std::array<std::uniform_real_distribution<double>, dim> dist{};
    double vol_box;
};

} // namespace proposals
} // namespace mc

#include "uniformProposal.tpp"

#endif // MONTECARLO_1_UNIFORM_PROPOSAL_HPP