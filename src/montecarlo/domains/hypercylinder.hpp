/**
 * @file hypercylinder.hpp
 * @brief N-dimensional hypercylinder (infinite cylinder in discrete dimensions).
 * @details Represents a generalization of a cylinder to N dimensions,
 * where an (N-1)-dimensional hypersphere is extruded along the N-th axis.
 */

#ifndef MONTECARLO_1_HYPERCYLINDER_HPP
#define MONTECARLO_1_HYPERCYLINDER_HPP

#include "integration_domain.hpp"
#include "../geometry.hpp"
#include <cmath>

namespace mc::domains {

/**
 * @brief N-dimensional hypercylinder.
 * @tparam dim Dimensionality (e.g., 2 for annulus, 3 for cylinder, N for hypercylinder).
 * 
 * Represents a solid hypercylinder defined as an (N-1)-dimensional hypersphere
 * of radius `r` extended (extruded) along the last dimension by height `h`:
 * C = {x ∈ ℝⁿ : x₁² + ... + x_{n-1}² ≤ r², 0 ≤ xₙ ≤ h}
 * 
 * Bounding box: [-r, r]^(N-1) × [0, h].
 */
template <size_t dim>
class HyperCylinder : public IntegrationDomain<dim> {
public:
    /**
     * @brief Construct a hypercylinder with given base radius and height.
     * @param rad Radius of the (N-1)-dimensional hypersphere base (must be > 0).
     * @param h   Height along the last dimension (must be ≥ 0).
     * @note Requires dim ≥ 2.
     */
    HyperCylinder(double rad, double h);

    /**
     * @brief Get the axis-aligned bounding box (hypercube).
     * @return Bounds [-radius, radius] for first N-1 dimensions,
     *         and [0, height] for the last dimension.
     */
    geom::Bounds<dim> getBounds() const override;

    /**
     * @brief Get the volume of the bounding box.
     * @return (2*radius)^(dim-1) * height
     * @details Used for Monte Carlo weight normalization in sampling.
     */
    double getBoxVolume() const override;

    /**
     * @brief Test if a point is inside the hypercylinder.
     * @param p Point to test.
     * @return true if radial distance² ≤ radius² AND 0 ≤ p[last] ≤ height;
     *         false otherwise.
     */
    bool isInside(const geom::Point<dim> &p) const override;

private:
    /** Radius of the hypersphere base (first N-1 dimensions). */
    double radius;
    /** Height along the last dimension. */
    double height;
};

// Ensure the implementation code is saved as "hypercylinder.tpp"

} // namespace mc::domains

#include "hypercylinder.tpp"

#endif //MONTECARLO_1_HYPERCYLINDER_HPP