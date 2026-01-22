// GA.hpp
#pragma once
#include "optimizer.hpp"
#include <vector>
#include <random>
#include <stdexcept>

namespace mc{
namespace optim{

    struct GAConfig {
        size_t population_size = 80;
        size_t max_generations = 200;

        size_t tournament_k = 3;

        Real crossover_rate = 0.9;
        Real mutation_rate  = 0.1;
        Real mutation_sigma = 0.1;

        size_t elitism_count = 1;
    };

    class GA : public Optimizer {
    public:
        struct Individual {
            Coordinates genome;
            Real fitness;
        };

        explicit GA(const GAConfig& config = GAConfig{});

        void setObjectiveFunction(ObjectiveFunction func) override;
        void setBounds(const Coordinates& lower, const Coordinates& upper) override;
        void setMode(OptimizationMode mode) override;
        void setCallback(StepCallback cb) override;

        Solution optimize() override;
        void step() override;
        [[nodiscard]] Solution getBestSolution() const override;

        [[nodiscard]] const std::vector<Individual>& getPopulation() const {
            return m_population;
        }

    private:
        void initialize();
        void evaluate(Individual& ind); // Thread-safe best update (critical)
        void enforceBounds(Coordinates& x);

        const Individual& tournamentSelect();
        void crossoverUniform(const Coordinates& p1, const Coordinates& p2,
                              Coordinates& c1, Coordinates& c2);
        void mutateGaussian(Coordinates& x);

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