/**
 * @file RngManager.hpp
 * @brief Thread-safe random number generator management
 * @author Giacomo Merlo
 * @date 12/01/26
 * 
 * Provides deterministic RNG seeding for parallel Monte Carlo simulations,
 * ensuring reproducibility across different thread counts and run configurations.
 */

#ifndef MONTECARLO_DGL_RNGMANAGER_HPP
#define MONTECARLO_DGL_RNGMANAGER_HPP

#include <random>
#include <cstdint>

/**
 * @brief Factory for creating deterministically seeded RNG instances
 * 
 * Creates independent Mersenne Twister generators for each thread/run combination
 * using a master seed. This ensures that parallel computations are reproducible
 * and that different threads don't produce correlated random sequences.
 * 
 * @note The seed sequence combines master_seed, thread_id, and run_id to guarantee
 *       statistical independence between generators.
 */
class RngManager {
public:
    /**
     * @brief Construct RNG manager with a master seed
     * @param seed Master seed for all derived generators
     */
    explicit RngManager(uint64_t seed) : master_seed(seed) {}

    /**
     * @brief Create a deterministic RNG for a specific thread/run
     * @param thread_id Unique identifier for the calling thread (e.g., from omp_get_thread_num())
     * @param run_id Optional run identifier for multiple independent executions (default: 0)
     * @return Mersenne Twister RNG seeded deterministically
     * 
     * @details The seed sequence is constructed from (master_seed, thread_id, run_id),
     *          ensuring that each thread gets a statistically independent random stream.
     *          Calling this method with the same parameters always produces the same sequence.
     */
    std::mt19937 make_rng(int thread_id, int run_id = 0) const {
        std::seed_seq seq{
            static_cast<uint32_t>(master_seed),
            static_cast<uint32_t>(thread_id),
            static_cast<uint32_t>(run_id)
        };
        return std::mt19937(seq);
    }

private:
    uint64_t master_seed; ///< Base seed for all derived generators
};

#endif //MONTECARLO_DGL_RNGMANAGER_HPP