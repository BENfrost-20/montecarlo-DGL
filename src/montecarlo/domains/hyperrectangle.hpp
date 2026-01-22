/**
 * @file hyperrectangle.hpp
 * @brief Axis-aligned hyperrectangular domain in N dimensions.
 * @details Implements a simple but efficient integration domain representing
 * a Cartesian box (hyperrectangle) centered at the origin with specified
 * extents along each dimension.
 */

#ifndef MONTECARLO_1_HYPERRECTANGLE_HPP
#define MONTECARLO_1_HYPERRECTANGLE_HPP

#include "integration_domain.hpp"
#include <utility>
#include "../geometry.hpp"

namespace mc::domains {

/**
 * @brief Axis-aligned hyperrectangular domain.
 * @tparam dim Dimensionality of the space.
 * 
 * Represents a Cartesian box centered at the origin:
 * [-d₁/2, d₁/2] × [-d₂/2, d₂/2] × ... × [-dₙ/2, dₙ/2]
 * where dᵢ is the extent along dimension i.
 * 
 * All containment checks and volume computations run in O(dim) time.
 */
template <size_t dim>
class HyperRectangle : public IntegrationDomain<dim>{
public:
    /**
     * @brief Construct a hyperrectangle from dimension extents.
     * @param dims Array of dimension sizes (extent along each axis).
     *             The box is centered at origin with ±dims[i]/2 bounds.
     */
    HyperRectangle(std::array<double, dim> &dims);

    /**
     * @brief Get the axis-aligned bounding box (coincides with the domain).
     * @return Bounds [-dims[i]/2, +dims[i]/2] for each dimension i.
     */
    mc::geom::Bounds<dim> getBounds() const override;

    /**
     * @brief Compute the volume of the hyperrectangle.
     * @return Product of all dimension extents: ∏ dims[i].
     */
    double getBoxVolume() const override;

    /**
     * @brief Test if a point is inside the hyperrectangle.
     * @param point Query point.
     * @return true if |point[i]| ≤ dims[i]/2 for all dimensions; false otherwise.
     */
    bool isInside(const mc::geom::Point<dim> &point) const override;
private:
    /** Extent (side length) along each axis. */
    std::array<double, dim> dimensions;
};

} // namespace mc::domains

#include "hyperrectangle.tpp"

#endif //MONTECARLO_1_HYPERRECTANGLE_HPP