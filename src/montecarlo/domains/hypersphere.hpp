/**
 * @file hypersphere.hpp
 * @brief N-dimensional hypersphere (solid ball) domain.
 * @details Implements a solid ball (closed sphere) of specified radius
 * centered at the origin. Supports arbitrary dimensions via template parameter.
 */

#ifndef HYPERSPHERE_HPP
#define HYPERSPHERE_HPP

#include "integration_domain.hpp"
#include <utility>
#include "../geometry.hpp"

namespace mc::domains {

/**
 * @brief N-dimensional ball (solid sphere).
 * @tparam dim Dimensionality (e.g., 2 for disk, 3 for ball, N for hypersphere).
 * 
 * Represents a closed ball of specified radius centered at the origin:
 * B = {x ∈ ℝⁿ : ||x|| ≤ r}
 * 
 * @details Uses the closed-form volume formula:
 * V = (π^(n/2) / Γ(n/2 + 1)) * r^n
 * 
 * Bounding box is a hypercube [-r, r]ⁿ.
 */
template <size_t dim>
class Hypersphere : public IntegrationDomain<dim>{
public:
    /**
     * @brief Construct a hypersphere of given radius.
     * @param rad Radius of the sphere (must be > 0).
     */
    Hypersphere(double rad);

    /**
     * @brief Get the axis-aligned bounding box (hypercube).
     * @return Bounds [-radius, radius] for each dimension.
     */
    mc::geom::Bounds<dim> getBounds() const override;

    /**
     * @brief Get the volume of the bounding hypercube.
     * @return (2*radius)^dim
     */
    double getBoxVolume() const override;

    /**
     * @brief Test if a point is inside the hypersphere.
     * @param point Point to test.
     * @return true if ||point|| ≤ radius; false otherwise.
     */
    bool isInside(const mc::geom::Point<dim> &point) const override;
private:
    /** Radius of the hypersphere. */
    double radius;
};

} // namespace mc::domains

#include "hypersphere.tpp"

#endif //HYPERSPHERE_HPP