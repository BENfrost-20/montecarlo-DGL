/**
 * @file hyperrectangle.hpp
 * @brief Axis-aligned hyperrectangular domain in N dimensions
 * 
 * Implements a simple but efficient integration domain representing
 * a Cartesian box defined by extents along each axis.
 */

#ifndef MONTECARLO_1_HYPERRECTANGLE_HPP
#define MONTECARLO_1_HYPERRECTANGLE_HPP

#include "integration_domain.hpp"
#include <utility>
#include "../geometry.hpp"

namespace mc::domains {

/**
 * @brief Axis-aligned hyperrectangular domain
 * @tparam dim Dimensionality
 * 
 * Represents a Cartesian box [a₁,b₁] × [a₂,b₂] × ... × [aₙ,bₙ]
 * defined by extents along each dimension. All containment checks
 * and volume computations run in O(dim) time.
 */
template <size_t dim>
class HyperRectangle : public IntegrationDomain<dim>{
public:
    /**
     * @brief Construct hyperrectangle from dimension extents
     * @param dims Array of dimension sizes (extent along each axis)
     */
    HyperRectangle(std::array<double, dim> &dims);

    /**
     * @brief Get bounding box (coincides with domain itself)
     * @return Bounds covering the entire hyperrectangle
     */
    mc::geom::Bounds<dim> getBounds() const override;

    /**
     * @brief Compute hyperrectangle volume
     * @return Product of all dimension extents
     */
    double getBoxVolume() const override;

    /**
     * @brief Test if point is inside hyperrectangle
     * @param point Query point
     * @return true if point ∈ [center - dims/2, center + dims/2] on all axes
     */
    bool isInside(const mc::geom::Point<dim> &point) const override;
private:
    std::array<double, dim> dimensions; ///< Extent along each axis
};

} // namespace mc::domains

#include "hyperrectangle.tpp"

#endif //MONTECARLO_1_HYPERRECTANGLE_HPP