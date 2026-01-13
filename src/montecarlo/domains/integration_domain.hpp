/**
 * @file integration_domain.hpp
 * @brief Abstract interface for N-dimensional integration domains
 * 
 * Defines the contract that all geometric domains must satisfy to support
 * Monte Carlo integration. Concrete implementations include hyperspheres,
 * hyperrectangles, cylinders, and arbitrary convex polytopes.
 */

#ifndef MONTECARLO_1_INTEGRATION_DOMAIN_HPP
#define MONTECARLO_1_INTEGRATION_DOMAIN_HPP

#include <array>
#include "../geometry.hpp"

using namespace std;
using namespace geom;

/**
 * @brief Abstract base class for N-dimensional integration domains
 * @tparam dim Dimensionality of the domain
 * 
 * Provides the interface required by Monte Carlo integrators to sample
 * and evaluate points within geometric regions. All domain types must
 * implement bounding box, volume, and point containment queries.
 * 
 * @note This is a pure virtual interface requiring concrete implementations.
 */
template <size_t dim>
class IntegrationDomain {
public:

    /**
     * @brief Get the axis-aligned bounding box of the domain
     * @return Bounds object containing min/max extents along each axis
     * 
     * Used by samplers to generate candidate points uniformly
     * within a hyperrectangle enclosing the actual domain.
     */
    // returns the boundaries o f each dimension of the integration domain
    virtual Bounds<dim> getBounds() const = 0;

    /**
     * @brief Compute the volume of the bounding box
     * @return Volume of the hyperrectangular bounding region
     * 
     * Required for acceptance-rejection sampling and Hit-or-Miss
     * volume estimation. Returns the product of extents along all axes.
     */
    // returns the volume of the integration domain
    virtual double getBoxVolume() const = 0;

    /**
     * @brief Test whether a point lies inside the domain
     * @param point Point to test
     * @return true if point ∈ domain, false otherwise
     * 
     * This is the core containment predicate used during Monte Carlo
     * sampling. Should be implemented efficiently as it's called millions of times.
     */
    // returns true if the given point is inside the integration domain
    virtual bool isInside(const Point<dim> &point) const = 0;

    /// Virtual destructor for proper cleanup of derived classes
    virtual ~IntegrationDomain() = default;

};

#endif //MONTECARLO_1_INTEGRATION_DOMAIN_HPP