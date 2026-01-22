/**
 * @file rng_factory.hpp
 * @brief Factory for creating deterministic, independent RNG engines
 * @date 2026-01-21
 *
 * Provides functions to create std::mt19937 engines that are:
 * - Deterministic: same inputs → same sequence
 * - Independent: different stream_id/thread → statistically independent sequences
 * - Reproducible: results are the same across runs with same global seed
 *
 * Uses splitmix64-style mixing for robust seed mixing, avoiding correlations.
 */

#ifndef MONTECARLO_RNG_FACTORY_HPP
#define MONTECARLO_RNG_FACTORY_HPP

#include <random>
#include <cstdint>
#include <optional>

namespace mc {
namespace rng {


/**
 * @brief Create a deterministic RNG engine for a specific stream
 * @param stream_id Unique identifier for this RNG stream (default: 0)
 * @return std::mt19937 engine seeded deterministically
 */
std::mt19937 make_engine(std::uint64_t stream_id = 0);

/**
 * @brief Create a deterministic RNG engine for the current thread (OpenMP if available)
 * @param stream_id Additional stream identifier (default: 0)
 * @return std::mt19937 engine seeded deterministically based on thread and stream
 *
 * If OpenMP is not enabled, falls back to thread_id = 0.
 */
std::mt19937 make_thread_engine(std::uint64_t stream_id = 0);

/**
 * @brief Create a deterministic RNG engine with explicit seed override (optional)
 * @param base_seed If provided, used instead of global seed; otherwise global seed is used
 * @param stream_id Unique identifier for this RNG stream
 * @return std::mt19937 engine seeded deterministically
 *
 * This avoids the anti-pattern "seed==0 means fallback", since 0 is a valid seed.
 */
std::mt19937 make_engine_with_seed(std::optional<std::uint32_t> base_seed,
                                   std::uint64_t stream_id);

} // namespace rng
} // namespace mc

#endif // MONTECARLO_RNG_FACTORY_HPP