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

        using StepCallback = std::function<void(const Solution& current_best, size_t iteration)>;

        virtual ~Optimizer() = default;

        // --- Configuration Methods ---

        /**
         * @brief Sets the function to optimize (the "black box").
         */
        virtual void setObjectiveFunction(ObjectiveFunction func) = 0;

        /**
         * @brief Defines the search space boundaries (hyper-rectangle).
         * @param lower_bounds Vector of minimum values for each dimension.
         * @param upper_bounds Vector of maximum values for each dimension.
         */
        virtual void setBounds(const Coordinates& lower_bounds, const Coordinates& upper_bounds) = 0;

        /**
         * @brief Sets the optimization goal (Minimization or Maximization).
         */
        virtual void setMode(OptimizationMode mode) = 0;

        virtual void setCallback(StepCallback cb) = 0;

        // --- Execution Methods ---

        /**
         * @brief Runs the optimization loop until the stopping criterion is met.
         * @return The best solution found.
         */
        virtual Solution optimize() = 0;

        /**
         * @brief Performs a single iteration/generation of the algorithm.
         * Useful for debugging, plotting, or GUI integration.
         */
        virtual void step() = 0;

        // --- Inspection Methods ---

        /**
         * @brief Returns the best solution found SO FAR.
         */
        [[nodiscard]] virtual Solution getBestSolution() const = 0;
    };
} //namespace mc
} //namespace optim