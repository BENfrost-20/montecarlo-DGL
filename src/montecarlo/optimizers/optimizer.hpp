/**
 * @file optimizer.hpp
 * @brief Abstract optimizer interface for PSO, GA, and future algorithms.
 * @details Defines the common lifecycle (configure, optimize, step, inspect) and
 * callback semantics used by all optimizers in this package.
 */
#pragma once
#include "types.hpp"

namespace mc{
namespace optim{

    /**
     * @brief Abstract base class for all optimization algorithms.
     * Enforces a common interface for GA, PSO, etc.
     */
    class Optimizer {
    public:

        /**
         * @brief Callback invoked after each step/generation.
         * @param current_best The best solution so far.
         * @param iteration Zero-based iteration/generation index.
         */
        using StepCallback = std::function<void(const Solution& current_best, size_t iteration)>;

        virtual ~Optimizer() = default;

        // --- Configuration Methods ---

        /**
         * @brief Set the function to optimize (the "black box").
         * @param func Objective mapping coordinates to scalar value.
         */
        virtual void setObjectiveFunction(ObjectiveFunction func) = 0;

        /**
         * @brief Define the search space boundaries (hyper-rectangle).
         * @param lower_bounds Minimum coordinate per dimension.
         * @param upper_bounds Maximum coordinate per dimension.
         */
        virtual void setBounds(const Coordinates& lower_bounds, const Coordinates& upper_bounds) = 0;

        /**
         * @brief Set the optimization goal.
         */
        virtual void setMode(OptimizationMode mode) = 0;

        virtual void setCallback(StepCallback cb) = 0;

        // --- Execution Methods ---

        /**
         * @brief Run the optimization loop until the stopping criterion is met.
         * @return The best solution found.
         */
        virtual Solution optimize() = 0;

        /**
         * @brief Perform a single iteration/generation of the algorithm.
         * Useful for debugging, plotting, or GUI integration.
         */
        virtual void step() = 0;

        // --- Inspection Methods ---

        /**
         * @brief Get the best solution found so far.
         */
        [[nodiscard]] virtual Solution getBestSolution() const = 0;
    };
} //namespace mc
} //namespace optim