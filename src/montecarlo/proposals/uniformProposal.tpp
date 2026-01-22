#ifndef MONTECARLO_1_UNIFORM_PROPOSAL_TPP
#define MONTECARLO_1_UNIFORM_PROPOSAL_TPP

namespace mc {
namespace proposals {

/**
 * @brief Construct a uniform proposal over a domain's bounding box.
 * @tparam dim Dimensionality parameter.
 * @param d Integration domain to sample from.
 * 
 * @details Initializes uniform distributions for each dimension spanning
 * the domain's bounding box. Precomputes the box volume for PDF evaluation.
 */
template <size_t dim>
UniformProposal<dim>::UniformProposal(const mc::domains::IntegrationDomain<dim>& d)
    : domain(d), vol_box(d.getBoxVolume())
{
    auto bounds = domain.getBounds();
    for (size_t i = 0; i < dim; ++i) {
        dist[i] = std::uniform_real_distribution<double>(
            bounds[i].first, bounds[i].second
        );
    }
}

/**
 * @brief Sample a uniform point from the domain's bounding box.
 * @tparam dim Dimensionality parameter.
 * @param rng Mersenne Twister random generator.
 * @return Point uniformly distributed in the bounding box.
 * 
 * @details Generates independent uniform samples along each dimension.
 * Time complexity: O(dim).
 */
template <size_t dim>
mc::geom::Point<dim> UniformProposal<dim>::sample(std::mt19937& rng) const
{
    mc::geom::Point<dim> x;
    for (size_t i = 0; i < dim; ++i) {
        x[i] = dist[i](rng);
    }
    return x;
}

/**
 * @brief Evaluate the uniform probability density.
 * @tparam dim Dimensionality parameter.
 * @param x Query point (ignored for uniform distribution).
 * @return 1 / V where V is the bounding box volume.
 * 
 * @details Constant probability density over the entire bounding box.
 * This PDF is well-defined only if x is inside the bounding box;
 * outside the box, the density is technically undefined (but we return
 * the constant value anyway for API consistency).
 */
template <size_t dim>
double UniformProposal<dim>::pdf(const mc::geom::Point<dim>&) const
{
    return 1.0 / vol_box;
}

} // namespace proposals
} // namespace mc

#endif // MONTECARLO_1_UNIFORM_PROPOSAL_TPP