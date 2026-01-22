/**
 * @file hypersphere.tpp
 * @brief Hypersphere template implementation.
 * @details Contains the inline implementations of `Hypersphere` methods
 * for N-dimensional ball volume and containment testing.
 */

#include <cmath>
#include "../geometry.hpp"
#include "integration_domain.hpp"

namespace mc::domains {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * @brief Construct a hypersphere of given radius.
 * @tparam dim Dimensionality of the space.
 * @param rad Radius of the hypersphere (must be > 0).
 * 
 * @details Initializes an N-dimensional solid ball centered at the origin
 * with the specified radius. The ball is closed: ||x|| ≤ rad.
 */
template <size_t dim>
Hypersphere<dim>::Hypersphere(double rad):
    radius(rad)
{}

/**
 * @brief Get the axis-aligned bounding box enclosing the hypersphere.
 * @tparam dim Dimensionality parameter.
 * @return Bounds with min/max = ±radius for all dimensions.
 * 
 * @details The minimal axis-aligned bounding box of a hypersphere
 * is a hypercube [-r, r]^dim. This is used for rejection sampling.
 */
template <size_t dim>
auto Hypersphere<dim>::getBounds() const -> mc::geom::Bounds<dim> {
    mc::geom::Bounds<dim> bounds;
    for(size_t i = 0; i < dim; ++i)
        bounds[i] = std::make_pair(-radius, radius);
    return bounds;
}

/**
 * @brief Compute the volume of the bounding hypercube.
 * @tparam dim Dimensionality parameter.
 * @return (2*radius)^dim
 * 
 * @details This is the volume of the minimal axis-aligned bounding box,
 * not the actual hypersphere volume. Used for normalization in Monte Carlo.
 * The actual ball volume is: V = (π^(n/2) / Γ(n/2 + 1)) * r^n.
 */
template <size_t dim>
double Hypersphere<dim>::getBoxVolume() const{
    return std::pow(2 * radius, dim);
}

/**
 * @brief Test whether a point lies inside the hypersphere.
 * @tparam dim Dimensionality parameter.
 * @param point Point to test.
 * @return true if ||point|| ≤ radius (point is in the closed ball).
 * 
 * @details Computes the Euclidean norm and tests against the radius.
 * Time complexity: O(dim) for norm computation.
 */
template <size_t dim>
bool Hypersphere<dim>::isInside(const mc::geom::Point<dim> &point) const{
    double norm = 0;
    for(size_t k = 0; k<dim; ++k){
        norm+=std::pow(point[k], 2);
    }
    return std::sqrt(norm) <= radius;
}

} // namespace mc::domains