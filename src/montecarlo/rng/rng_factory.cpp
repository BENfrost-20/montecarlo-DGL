/**
 * @file rng_factory.cpp
 * @brief Implementation of RNG factory functions
 * @author Generated for montecarlo-DGL refactoring
 * @date 2026-01-21
 * 
 * Uses splitmix64 for robust seed mixing, as recommended by the Mersenne Twister
 * authors and used in many high-quality RNG implementations.
 * 
 * DESIGN RATIONALE:
 * 
 * 1. **splitmix64 mixing**: Combines multiple inputs (seed, stream_id, thread_id)
 *    into a single high-quality seed. This avoids correlations that can occur
 *    when simply adding or XORing seeds together.
 * 
 * 2. **No shared state**: Each call creates a fresh engine. No static mt19937
 *    is shared between threads, avoiding data races.
 * 
 * 3. **Determinism**: Same inputs always produce the same RNG state, enabling
 *    reproducible simulations.
 * 
 * 4. **Thread safety**: The only shared state (global seed) is atomic.
 *    All other operations are thread-local.
 */

#include "rng_factory.hpp"
#include "rng_global.hpp"
#include <omp.h>

namespace mc {

namespace {

/**
 * @brief splitmix64 mixing function
 * 
 * This is a high-quality 64-bit mixer based on the splitmix64 PRNG by
 * Sebastiano Vigna. It's used to combine multiple seed components into
 * a single high-entropy seed.
 * 
 * Properties:
 * - Bijective (reversible)
 * - Excellent avalanche characteristics
 * - Fast (a few CPU cycles)
 * 
 * Reference: https://prng.di.unimi.it/splitmix64.c
 */
inline std::uint64_t splitmix64_mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

/**
 * @brief Combine base seed with stream_id and optional thread_id
 * @param base_seed The base seed (usually from get_global_seed())
 * @param stream_id Stream identifier for this RNG instance
 * @param thread_id Thread identifier (0 for non-threaded use)
 * @return A well-mixed 64-bit seed
 * 
 * The mixing formula ensures that small changes in any input produce
 * dramatically different outputs, preventing correlated RNG streams.
 */
inline std::uint64_t combine_seeds(std::uint64_t base_seed, 
                                   std::uint64_t stream_id, 
                                   std::uint64_t thread_id) {
    // Golden ratio constant for additional mixing
    constexpr std::uint64_t GOLDEN = 0x9e3779b97f4a7c15ULL;
    
    // Combine all components with offsets to avoid zero-related issues
    std::uint64_t combined = base_seed;
    combined ^= splitmix64_mix(stream_id + GOLDEN);
    combined ^= splitmix64_mix((thread_id << 1) + 1);  // +1 to handle thread_id=0
    
    // Final mix to ensure good distribution
    return splitmix64_mix(combined);
}

/**
 * @brief Create mt19937 from a 64-bit seed
 * 
 * Uses std::seed_seq to properly initialize all 624 state words
 * of the Mersenne Twister from our 64-bit mixed seed.
 */
inline std::mt19937 create_mt19937(std::uint64_t seed) {
    // Split 64-bit seed into two 32-bit values for seed_seq
    std::uint32_t lo = static_cast<std::uint32_t>(seed & 0xFFFFFFFFULL);
    std::uint32_t hi = static_cast<std::uint32_t>(seed >> 32);
    
    std::seed_seq seq{lo, hi};
    return std::mt19937(seq);
}

} // anonymous namespace


std::mt19937 make_engine(std::uint64_t stream_id) {
    std::uint64_t base = static_cast<std::uint64_t>(get_global_seed());
    std::uint64_t mixed = combine_seeds(base, stream_id, 0);
    return create_mt19937(mixed);
}

std::mt19937 make_thread_engine(std::uint64_t stream_id) {
    std::uint64_t base = static_cast<std::uint64_t>(get_global_seed());
    std::uint64_t tid = static_cast<std::uint64_t>(omp_get_thread_num());
    std::uint64_t mixed = combine_seeds(base, stream_id, tid);
    return create_mt19937(mixed);
}

std::mt19937 make_engine_with_seed(std::uint32_t base_seed, std::uint64_t stream_id) {
    // If base_seed is 0, use global seed (backward compatibility)
    std::uint64_t base = (base_seed == 0) 
        ? static_cast<std::uint64_t>(get_global_seed())
        : static_cast<std::uint64_t>(base_seed);
    
    std::uint64_t mixed = combine_seeds(base, stream_id, 0);
    return create_mt19937(mixed);
}

} // namespace mc
