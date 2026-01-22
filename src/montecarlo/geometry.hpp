/**
 * @file geometry.hpp
 * @brief Core geometric types for Monte Carlo integration
 * 
 * Defines fundamental geometric primitives including points and bounding boxes
 * in N-dimensional space for use throughout the Monte Carlo library.
 */

#ifndef MONTECARLO_1_GEOMETRY_HPP
#define MONTECARLO_1_GEOMETRY_HPP


#include <array>
#include <cstddef>

/**
 * @namespace geom
 * @brief Geometric types and utilities for N-dimensional spaces.
 * Contains fundamental primitives like Point and Bounds used throughout the Monte Carlo library.
 */
namespace mc::geom {

/**
 * @brief N-dimensional point representation
 * @tparam dim Dimensionality of the point
 * 
 * Provides a lightweight container for coordinates in dim-dimensional space.
 * Supports index-based access to individual coordinates.
 */
template <int dim>
class Point
{
public:
    /// Default constructor initializing all coordinates to 0.0
    Point() { coords.fill(0.0); }

    /**
     * @brief Non-const coordinate accessor
     * @param i Coordinate index (0 to dim-1)
     * @return Reference to the i-th coordinate
     */
    double& operator[](std::size_t i)       { return coords[i]; }
    
    /**
     * @brief Const coordinate accessor
     * @param i Coordinate index (0 to dim-1)
     * @return Value of the i-th coordinate
     */
    double  operator[](std::size_t i) const { return coords[i]; }

    /**
     * @brief Get the dimensionality of the point
     * @return Number of dimensions
     */
    std::size_t dimension() const { return dim; }

private:
    std::array<double, dim> coords; ///< Coordinate storage
};

/**
 * @brief N-dimensional axis-aligned bounding box
 * @tparam dim Dimensionality of the bounds
 * 
 * Represents a hyperrectangular region defined by min/max pairs along each axis.
 * Used for defining integration domains and sampling regions.
 */
template <int dim>
class Bounds
{
public:
    /**
     * @brief Default constructor initializing all bounds to [0,0]
     */
    Bounds()
    {
        for (std::size_t i = 0; i < dim; ++i)
        {
            bounds[i] = std::make_pair(0.0, 0.0);
        }
    }

    /**
     * @brief Const accessor for bounds along axis i
     * @param i Axis index (0 to dim-1)
     * @return Pair (min, max) defining the extent along axis i
     */
    const std::pair<double, double> &operator[](std::size_t i) const 
    {
        return bounds[i];
    }

    /**
     * @brief Non-const accessor for bounds along axis i
     * @param i Axis index (0 to dim-1)
     * @return Reference to pair (min, max) defining the extent along axis i
     */
    std::pair<double, double> &operator[](std::size_t i)
    {
        return bounds[i];
    }

private:
    std::array<std::pair<double, double>, dim> bounds; ///< Min/max pairs for each dimension
};
                                                                                        

} // namespace mc::geom

#endif // MONTECARLO_1_GEOMETRY_HPP