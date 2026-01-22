// GA.hpp
/**
 * @file GA.hpp
 * @brief Genetic Algorithm (GA) interface and data structures.
 * @details Implements tournament selection, uniform crossover and Gaussian
 * mutation with elitism. Evaluations may be parallelized; global-best updates
 * are performed serially to avoid races and ensure deterministic results.
 */
#pragma once
#include "optimizer.hpp"
#include <vector>
#include <random>
#include <stdexcept>

namespace mc{
namespace optim{

    /**
     * @brief Configuration parameters for GA.
     */
    struct GAConfig {
        /** Size of the population. */
        size_t population_size = 80;
        /** Number of generations to evolve. */
        size_t max_generations = 200;

        /** Tournament size for selection (k >= 2). */
        size_t tournament_k = 3;

        /** Probability of performing crossover in reproduction. */
        Real crossover_rate = 0.9;
        /** Per-gene mutation probability. */
        Real mutation_rate  = 0.1;
        /** Mutation magnitude (scaled by coordinate span). */
        Real mutation_sigma = 0.1;

        /** Number of top individuals copied unchanged to next generation. */
        size_t elitism_count = 1;
    };

    /**
     * @brief Genetic Algorithm optimizer.
     * @note Determinism: selection/variation use a shared RNG; fitness evaluation
     * can be parallelized. Best-solution tracking is updated serially.
     */
    class GA : public Optimizer {
    public:
        /**
         * @brief A single population member.
         */
        struct Individual {
            /** Encoded parameters (genome). */
            Coordinates genome;
            /** Fitness value for this genome. */
            Real fitness;
        };

        /**
         * @brief Construct a GA optimizer with the given configuration.
         */
        explicit GA(const GAConfig& config = GAConfig{});

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
         * @brief Register a callback invoked after each generation.
         * @param cb Receives current global-best and generation index.
         */
        void setCallback(StepCallback cb) override;

        /**
         * @brief Run GA for `max_generations`.
         * @return The best solution found.
         */
        Solution optimize() override;
        /**
         * @brief Perform one generation: selection, crossover, mutation, evaluation,
         * and serial update of the global best.
         */
        void step() override;
        /**
         * @brief Get the best solution found so far.
         */
        [[nodiscard]] Solution getBestSolution() const override;

        /**
         * @brief Access the current population.
         */
        [[nodiscard]] const std::vector<Individual>& getPopulation() const {
            return m_population;
        }

    private:
        /**
         * @brief Initialize population genomes within bounds and evaluate fitness.
         */
        void initialize();
        /**
         * @brief Compute fitness for an individual. No global-best update here;
         * best selection is performed serially to ensure determinism.
         */
        void evaluate(Individual& ind);
        /**
         * @brief Clamp genome coordinates to bounds.
         */
        void enforceBounds(Coordinates& x);

        /**
         * @brief Tournament selection among `tournament_k` random individuals.
         */
        const Individual& tournamentSelect();
        /**
         * @brief Uniform crossover: swap genes with 50% probability per dimension.
         */
        void crossoverUniform(const Coordinates& p1, const Coordinates& p2,
                              Coordinates& c1, Coordinates& c2);
        /**
         * @brief Gaussian mutation applied per gene with probability `mutation_rate`.
         * Magnitude scaled by `mutation_sigma` times coordinate span.
         */
        void mutateGaussian(Coordinates& x);

        /**
         * @brief Compare raw fitness values according to optimization mode.
         */
        bool isBetterFitness(Real a, Real b) const;

        GAConfig m_config;
        OptimizationMode m_mode = OptimizationMode::MINIMIZE;
        ObjectiveFunction m_func;

        Coordinates m_lower_bounds;
        Coordinates m_upper_bounds;

        std::vector<Individual> m_population;
        Solution m_global_best;

        std::mt19937 m_rng;
        bool m_initialized = false;

        StepCallback m_callback;
        size_t m_generation = 0;
    };

} //namespace mc
} //namespace optim