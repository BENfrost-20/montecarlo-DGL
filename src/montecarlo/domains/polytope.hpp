/**
 * @file polytope.hpp
 * @brief N-dimensional polytope (convex polyhedron) domain for Monte Carlo integration.
 * @details A polytope is defined as the intersection of half-spaces, specified by vertices
 * and the inward-pointing normal vectors and offsets of its facets (hyperplane constraints).
 * This allows representation of arbitrary convex polyhedral regions in N dimensions.
 */

#ifndef MONTECARLO_1_POLYTOPE_H
#define MONTECARLO_1_POLYTOPE_H

#include "integration_domain.hpp"
#include "../geometry.hpp"
#include <vector>
#include <stdexcept>

namespace mc::domains {

/**
 * @class PolyTope
 * @brief Convex polytope (convex polyhedron) integration domain.
 * @tparam dim The dimensionality of the space.
 * 
 * A polytope is the intersection of half-spaces defined by linear inequalities:
 * normal_i · point + offset_i ≤ 0  for all facets i
 * 
 * where normal_i is an inward-pointing unit normal and offset_i is the constant term.
 * This representation supports arbitrary convex polytopes with any number of facets.
 * 
 * @note For efficiency, ensure normals are normalized and offsets are accurate.
 * @note Typically generated using Qhull (Qt Qx Fn flags).
 */
template <size_t dim>
class PolyTope : public IntegrationDomain<dim> {
public:

    /**
     * @brief Construct a polytope from vertices and facet constraints.
     * @param vertices Vector of vertices (corners) of the polytope.
     *                 Used only for computing the bounding box.
     * @param norms Vector of inward-pointing normal vectors; one per facet.
     *              Each normal is an array of dim coefficients.
     * @param offs Vector of offset values (constant terms); one per facet.
     * @throws std::runtime_error if norms and offs have different sizes,
     *         or if vertices is empty.
     * @note norms.size() must equal offs.size() (number of facets).
     */
    PolyTope(const std::vector<geom::Point<dim>>&   vertices,
             const std::vector<std::array<double, dim>>&  norms,
             const std::vector<double>&      offs);

    /**
     * @brief Get the axis-aligned bounding box containing the polytope.
     * @return Bounds<dim> with min/max coordinates in each dimension,
     *         computed from the vertex set.
     */
    geom::Bounds<dim> getBounds() const override;

    /**
     * @brief Get the volume of the axis-aligned bounding box.
     * @return Product of side lengths: ∏(max_i - min_i)
     * @details Used for Monte Carlo weight normalization in sampling.
     */
    double getBoxVolume() const override;

    /**
     * @brief Test if a point is inside or on the boundary of the polytope.
     * @param p The point to test.
     * @return true if normal_i · p + offset_i ≤ (tol) for all facets i;
     *         false otherwise (point violates at least one constraint).
     * @note Applies a small numerical tolerance (1e-12) for robustness.
     */
    bool isInside(const geom::Point<dim> &p) const override;

private:
    /** Vertices (corner points) of the polytope. Used for bounding box computation. */
    std::vector<mc::geom::Point<dim>> vec;
    /** Inward-pointing normal vectors of facets. One per facet, each of size dim. */
    std::vector<std::array<double, dim>> normals;
    /** Offset values (constant terms) of facet inequalities. One per facet. */
    std::vector<double> offsets;
};

} // namespace mc::domains

#include "polytope.tpp"

#endif //MONTECARLO_1_POLYTOPE_H