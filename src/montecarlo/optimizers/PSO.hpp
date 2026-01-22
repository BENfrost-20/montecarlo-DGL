/**
 * @file PSO.hpp
 * @brief Particle Swarm Optimization (PSO) interface and data structures.
 * @details Provides the `mc::optim::PSO` optimizer which minimizes or maximizes
 * a user-provided objective function within bounded coordinates. This
 * implementation supports OpenMP for parallel particle updates and ensures
 * deterministic behavior by decoupling random number generation from thread
 * scheduling. See PSO.cpp for implementation details.
 */
#pragma once
#include "optimizer.hpp"
#include <vector>
#include <random>

namespace mc{
namespace optim{

    /**
     * @brief Configuration parameters for PSO.
     */
    struct PSOConfig {
        /** Number of particles in the swarm. */
        size_t population_size = 50;
        /** Number of iterations to run the optimizer. */
        size_t max_iterations = 100;

        /** Inertia weight (w): scales previous velocity. */
        Real inertia_weight = 0.7;
        /** Cognitive coefficient (c1): scales attraction to particle best. */
        Real cognitive_coeff = 1.5;
        /** Social coefficient (c2): scales attraction to global best. */
        Real social_coeff = 1.5;
    };

    /**
     * @brief Particle Swarm Optimization algorithm.
     * @note Thread-safety: particle updates are parallelized; global-best updates are
     * performed serially to avoid races and ensure reproducibility.
     */
    class PSO : public Optimizer {
    public:
        /**
         * @brief A single particle in the swarm.
         */
        struct Particle {
            /** Current position in the search space. */
            Coordinates position;
            /** Current velocity vector. */
            Coordinates velocity;

            /** Best position found by this particle. */
            Coordinates best_position;
            /** Best objective value found by this particle. */
            Real best_value;
            /** Objective value at the current position. */
            Real current_value;
        };

        /**
         * @brief Construct a PSO optimizer with the given configuration.
         */
        explicit PSO(const PSOConfig& config = PSOConfig{});

        /**
         * @brief Set the objective function to optimize.
         * @param func Function mapping coordinates to scalar cost/fitness.
         */
        void setObjectiveFunction(ObjectiveFunction func) override;
        /**
         * @brief Set lower/upper bounds of the search hyper-rectangle.
         * @throws std::invalid_argument if dimensions mismatch.
         */
        void setBounds(const Coordinates& lower, const Coordinates& upper) override;
        /**
         * @brief Set optimization mode (minimize or maximize).
         */
        void setMode(OptimizationMode mode) override;
        /**
         * @brief Register a callback invoked after each `step()`.
         * @param cb Receives current global-best and iteration index.
         */
        void setCallback(StepCallback cb) override;

        /**
         * @brief Execute the optimization loop for `max_iterations`.
         * @return The best solution found.
         */
        Solution optimize() override;
        /**
         * @brief Perform one PSO iteration: update velocity/position, evaluate,
         * update personal and global bests.
         */
        void step() override;
        /**
         * @brief Get the best solution found so far.
         */
        [[nodiscard]] Solution getBestSolution() const override;

        /**
         * @brief Access the current swarm state (particles).
         */
        [[nodiscard]] const std::vector<Particle>& getParticles() const {
            return m_swarm;
        }

    private:
        /**
         * @brief Lazy initialization of swarm based on bounds and objective.
         */
        void initialize();
        /**
         * @brief Clamp particle positions to bounds and damp velocity on collision.
         */
        void enforceBounds(Particle& p);

        PSOConfig m_config;
        OptimizationMode m_mode = OptimizationMode::MINIMIZE;
        ObjectiveFunction m_func;

        Coordinates m_lower_bounds;
        Coordinates m_upper_bounds;

        std::vector<Particle> m_swarm;
        Solution m_global_best;

        bool m_initialized = false;

        /**
         * @brief Iteration counter used to derive unique RNG stream IDs per step
         * for deterministic behavior across different thread counts.
         */
        size_t m_current_iter = 0;

        StepCallback m_callback;
    };
} //namespace mc
} //namespace optim