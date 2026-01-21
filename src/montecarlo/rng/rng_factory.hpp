/**
 * @file rng_factory.hpp
 * @brief Factory for creating deterministic, independent RNG engines
 * @author Generated for montecarlo-DGL refactoring
 * @date 2026-01-21
 * 
 * Provides functions to create std::mt19937 engines that are:
 * - Deterministic: same inputs → same sequence
 * - Independent: different stream_id/thread → statistically independent sequences
 * - Reproducible: results are the same across runs with same global seed
 * 
 * Uses splitmix64 for robust seed mixing, avoiding correlations between streams.
 * 
 * USAGE:
 * @code
 *   // In serial code:
 *   auto rng = mc::make_engine(0);  // stream 0
 *   
 *   // In OpenMP parallel region:
 *   #pragma omp parallel
 *   {
 *       auto rng = mc::make_thread_engine(stream_id);
 *       // Each thread gets independent, reproducible RNG
 *   }
 * @endcode
 * 
 * @note For reproducibility in OpenMP, always use the same number of threads
 *       and call set_global_seed() before entering parallel regions.
 */

#ifndef MONTECARLO_RNG_FACTORY_HPP
#define MONTECARLO_RNG_FACTORY_HPP

#include <random>
#include <cstdint>

namespace mc {

/**
 * @brief Create a deterministic RNG engine for a specific stream
 * @param stream_id Unique identifier for this RNG stream (default: 0)
 * @return std::mt19937 engine seeded deterministically
 * 
 * The engine is seeded by mixing:
 * - The global seed (from get_global_seed())
 * - The stream_id
 * 
 * This ensures that:
 * - Different stream_ids produce statistically independent sequences
 * - Same global_seed + stream_id always produces the same sequence
 * 
 * Use different stream_id values for different logical purposes (e.g.,
 * initialization vs. iteration, or different particles in PSO).
 */
std::mt19937 make_engine(std::uint64_t stream_id = 0);

/**
 * @brief Create a deterministic RNG engine for the current thread
 * @param stream_id Additional stream identifier (default: 0)
 * @return std::mt19937 engine seeded deterministically based on thread and stream
 * 
 * The engine is seeded by mixing:
 * - The global seed (from get_global_seed())
 * - The stream_id
 * - The current thread ID (from omp_get_thread_num())
 * 
 * Use this in OpenMP parallel regions to ensure each thread has an
 * independent RNG that produces the same sequence across runs.
 * 
 * @note For reproducibility, ensure the same number of OpenMP threads
 *       are used across runs (e.g., via OMP_NUM_THREADS environment variable).
 */
std::mt19937 make_thread_engine(std::uint64_t stream_id = 0);

/**
 * @brief Create a deterministic RNG engine with explicit seed override
 * @param base_seed Seed to use instead of global seed
 * @param stream_id Unique identifier for this RNG stream
 * @return std::mt19937 engine seeded deterministically
 * 
 * Use this when you need to override the global seed for a specific operation,
 * for example when an API accepts an optional seed parameter.
 * 
 * If base_seed == 0, falls back to using the global seed.
 */
std::mt19937 make_engine_with_seed(std::uint32_t base_seed, std::uint64_t stream_id);

} // namespace mc

#endif // MONTECARLO_RNG_FACTORY_HPP
