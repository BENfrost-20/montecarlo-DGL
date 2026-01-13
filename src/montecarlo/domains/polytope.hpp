/**
 * @file polytope.hpp
 * @brief N-dimensional polytope (convex polyhedron) domain for Monte Carlo integration.
 * @details A polytope is defined as the intersection of half-spaces, specified by
 * vertices and the normal vectors and offsets of the facets (faces).
 * This allows representation of arbitrary convex polyhedral regions in N dimensions.
 */

#ifndef MONTECARLO_1_POLYTOPE_H
#define MONTECARLO_1_POLYTOPE_H

#include "integration_domain.hpp"
#include "../geometry.hpp"
#include <vector>

using namespace geom;
using namespace std;

/**
 * @class PolyTope
 * @brief Polytope (convex polyhedron) integration domain.
 * @tparam dim The dimensionality of the space.
 * @details A polytope is the intersection of half-spaces defined by:
 * normal_i · point + offset_i ≤ 0
 * where normal_i is the inward-pointing normal and offset_i is the constant term.
 * This template supports arbitrary convex polytopes with any number of facets.
 */
template <size_t dim>
class PolyTope : public IntegrationDomain<dim> {
public:

    /**
     * @brief Constructs a polytope from vertices and facet definitions.
     * @param vertices The vertices (corners) of the polytope.
     * @param norms Vector of inward-pointing normal vectors for each facet.
     * @param offs Offset values (constant terms) for each facet inequality.
     * @note norms and offs must have the same size (number of facets).
     */
    PolyTope(const std::vector<geom::Point<dim>>&   vertices,
                        const std::vector<array<double, dim>>&  norms,
                        const std::vector<double>&      offs);

    /**
     * @brief Returns the axis-aligned bounding box containing the polytope.
     * @return Bounds<dim> with min/max coordinates in each dimension.
     */
    geom::Bounds<dim> getBounds() const override;

    /**
     * @brief Returns the volume of the bounding box.
     * @return The product of side lengths: ∏(max_i - min_i).
     * @note Used for Monte Carlo weight normalization.
     */
    double getBoxVolume() const override;

    /**
     * @brief Checks if a point is inside or on the boundary of the polytope.
     * @param p The point to test.
     * @return true if normal_i · p + offset_i ≤ 0 for all facets; false otherwise.
     */
    bool isInside(const geom::Point<dim> &p) const override;

private:
    vector<Point<dim>> vec;           ///< Vertices of the polytope.
    vector<array<double, dim>> normals; ///< Inward-pointing normal vectors of facets.
    vector<double> offsets;           ///< Offset values for facet inequalities.
};

#include "polytope.tpp"

#endif //MONTECARLO_1_POLYTOPE_H