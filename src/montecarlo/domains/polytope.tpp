/**
 * @file polytope.tpp
 * @brief PolyTope template implementation.
 * @details Contains the inline implementations of `PolyTope` methods for
 * half-space intersection bounding box and point containment testing.
 * 
 * @note Qhull usage: `qhull Qt Qx Fn < points.txt > hull.txt`
 *       Points file format:  First line: <num_points> <dim>
 *                            Following lines: coordinates of each vertex
 */

#ifndef MONTECARLO_1_POLYTOPE_TPP
#define MONTECARLO_1_POLYTOPE_TPP

#include <cmath>
#include <algorithm> // For std::pow
#include "../geometry.hpp"
#include "integration_domain.hpp"
#include <vector>

namespace mc::domains {

template <size_t dim>
PolyTope<dim>::PolyTope(const std::vector<mc::geom::Point<dim>>&   vertices,
                        const std::vector<std::array<double, dim>>&  norms,
                        const std::vector<double>&      offs)
    : vec(vertices)
    , normals(norms)
    , offsets(offs)
{
    if (normals.size() != offsets.size()) {
        throw std::runtime_error(
            "PolyTope: normals and offsets must have the same size.");
    }

    if (vec.empty()) {
        throw std::runtime_error(
            "PolyTope: vertices list cannot be empty.");
    }
}

template<size_t dim>
mc::geom::Bounds<dim> PolyTope<dim>::getBounds() const{
    mc::geom::Bounds<dim> bounds;
    for (int i = 0; i < dim; ++i) {
        double max = vec[0][i];
        double min = vec[0][i];
        for (mc::geom::Point<dim> p: vec) {
            if (p[i] > max) max = p[i];
            if (p[i] < min) min = p[i];
        }
        bounds[i] = std::make_pair(min, max);
    }
    return bounds;
}

template<size_t dim>
double PolyTope<dim>::getBoxVolume() const {
    mc::geom::Bounds<dim> bou = this->getBounds();
    double vol = 1;
    for (int i = 0; i <dim; ++i) {
        vol *= (bou[i].second - bou[i].first);
    }
    return vol;
}

template<size_t dim>
bool PolyTope<dim>::isInside(const mc::geom::Point<dim> &point) const {
    const double tol = 1e-12; // Numerical tolerance for robust testing
    for (size_t i = 0; i < normals.size(); ++i) {
        double s = 0.0;
        // Compute dot product: normal_i · point
        for (size_t k = 0; k < dim; ++k)
            s += normals[i][k] * point[k];

        // Check half-space inequality: n_i · x + offset_i ≤ 0
        if (s > offsets[i] + tol)
            return false;  // Point violates this facet constraint
    }
    return true;
}

} // namespace mc::domains

#endif //MONTECARLO_1_POLYTOPE_TPP