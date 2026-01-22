/**
 * @file hyperrectangle.tpp
 * @brief HyperRectangle template implementation.
 * @details Contains the inline implementations of `HyperRectangle` methods
 * for axis-aligned box volume and containment testing.
 */

#include <cmath>
#include "../geometry.hpp"
#include "integration_domain.hpp"
#include <array>
#include <utility>

namespace mc::domains {

/**
 * @brief Construct an axis-aligned hyperrectangle from dimension extents.
 * @tparam dim Dimensionality of the space.
 * @param dims Array of dimension sizes. Box is centered at origin with ±dims[i]/2 bounds.
 * 
 * @details The resulting box spans [-d₁/2, d₁/2] × ... × [-dₙ/2, dₙ/2]
 * where dᵢ = dims[i].
 */
template <size_t dim>
HyperRectangle<dim>::HyperRectangle(std::array<double, dim> &dims):
    dimensions(dims)
{};

/**
 * @brief Get the axis-aligned bounding box (which coincides with the domain).
 * @tparam dim Dimensionality parameter.
 * @return Bounds [-dims[i]/2, dims[i]/2] for each dimension i.
 * 
 * @details For a hyperrectangle, the bounding box is exact (not approximation).
 */
template <size_t dim>
auto HyperRectangle<dim>::getBounds() const -> mc::geom::Bounds<dim> {
    mc::geom::Bounds<dim> bounds;
    for(size_t i = 0; i < dim; ++i)
        bounds[i] = std::make_pair(-dimensions[i]/2, dimensions[i]/2);
    return bounds;
}

/**
 * @brief Compute the volume of the hyperrectangle.
 * @tparam dim Dimensionality parameter.
 * @return Product of all dimension extents: ∏ dims[i].
 * 
 * @details Uses straightforward multiplication across all dimensions.
 * Time complexity: O(dim).
 */
template <size_t dim>
double HyperRectangle<dim>::getBoxVolume() const{
    double volume = 1;
    for(size_t i = 0; i < dim; ++i) {
        volume = volume * dimensions[i];
    }
    return volume;
}

/**
 * @brief Test whether a point lies inside the hyperrectangle.
 * @tparam dim Dimensionality parameter.
 * @param point Point to test.
 * @return true if |point[i]| ≤ dims[i]/2 for all dimensions; false otherwise.
 * 
 * @details Simple componentwise range checking. Returns true only if
 * the point is within the box on all dimensions.
 * Time complexity: O(dim).
 */
template <size_t dim>
bool HyperRectangle<dim>::isInside(const mc::geom::Point<dim> &point) const{
    bool inside = true;
        for(size_t i = 0; i < dim; ++i) {
            if (point[i] < -dimensions[i]/2 || point[i] > dimensions[i]/2) {
                inside = false;
            }
        }
    return inside;
}

} // namespace mc::domains