#pragma once
#include <vector>
#include <functional>
#include <limits>
#include <iostream>

namespace optimizers {

    // Define the precision used across the library.
    // Changing this to 'float' updates the whole library automatically.
    using Real = double;

    // A point in the N-dimensional search space.
    using Coordinates = std::vector<Real>;

    // The function signature for the problem to be solved.
    // Takes Coordinates as input, returns a scalar value (cost/fitness).
    using ObjectiveFunction = std::function<Real(const Coordinates&)>;

    // Enum to define the goal of the optimization.
    enum class OptimizationMode {
        MINIMIZE,
        MAXIMIZE
    };

    /**
     * @brief Represents a candidate solution in the search space.
     * Contains both the parameters (position) and the evaluated cost (value).
     */
    struct Solution {
        Coordinates params;
        Real value;

        // Helper to create a "worst-case" solution for initialization.
        // If minimizing, the worst value is Infinity. If maximizing, it's -Infinity.
        static Solution make_worst(OptimizationMode mode) {
            if (mode == OptimizationMode::MINIMIZE) {
                return { {}, std::numeric_limits<Real>::infinity() };
            } else {
                return { {}, -std::numeric_limits<Real>::infinity() };
            }
        }

        // Overload < operator for easy sorting/comparison
        bool isBetterThan(const Solution& other, OptimizationMode mode) const {
            if (mode == OptimizationMode::MINIMIZE) return value < other.value;
            return value > other.value;
        }
    };
}