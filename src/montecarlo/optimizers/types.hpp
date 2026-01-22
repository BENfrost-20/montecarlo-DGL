#pragma once
#include <vector>
#include <functional>
#include <limits>
#include <iostream>

/**
 * @file types.hpp
 * @brief Core optimizer type definitions for the Monte Carlo toolkit.
 * @details Collects shared aliases and data structures used by PSO and GA implementations.
 */

/**
 * @namespace optimizers
 * @brief Optimization framework with PSO, GA, and extensible Optimizer interface.
 * Contains type definitions, abstract optimizer base class, and concrete implementations
 * for particle swarm and genetic algorithm optimization.
 */
namespace mc{
namespace optim{

    /**
     * @brief Scalar precision used across optimizers.
     * @note Changing to `float` or long double updates the whole package.
     */
    using Real = double;

    /**
     * @brief A point in the N-dimensional search space.
     */
    using Coordinates = std::vector<Real>;

    /**
     * @brief Objective function signature.
     * @details Takes coordinates as input and returns a scalar cost/fitness.
     */
    using ObjectiveFunction = std::function<Real(const Coordinates&)>;

    /**
     * @brief Optimization goal.
     */
    enum class OptimizationMode {
        MINIMIZE,
        MAXIMIZE
    };

    /**
     * @brief Represents a candidate solution in the search space.
     * Contains both the parameters (position) and the evaluated cost (value).
     */
    struct Solution {
        /** Parameter vector (coordinates in the search space). */
        Coordinates params;
        /** Evaluated objective value for `params`. */
        Real value;

        /**
         * @brief Helper to create a worst-case solution for initialization.
         * @details If minimizing, the worst value is +Infinity; if maximizing,
         * it's -Infinity.
         */
        static Solution make_worst(OptimizationMode mode) {
            if (mode == OptimizationMode::MINIMIZE) {
                return { {}, std::numeric_limits<Real>::infinity() };
            } else {
                return { {}, -std::numeric_limits<Real>::infinity() };
            }
        }

        /**
         * @brief Compare two solutions according to the optimization mode.
         * @return true if this solution is better than `other`.
         */
        bool isBetterThan(const Solution& other, OptimizationMode mode) const {
            if (mode == OptimizationMode::MINIMIZE) return value < other.value;
            return value > other.value;
        }
    };
} //namespace mc
} //namespace optim