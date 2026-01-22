/**
 * @file rng_factory.cpp
 * @brief Implementation of RNG factory functions
 * @date 2026-01-21
 *
 * DESIGN:
 * - splitmix64-style mixing for strong avalanche between (seed, stream_id, thread_id)
 * - no shared RNG state
 * - deterministic and thread-safe (global seed is atomic in rng_global)
 * - OpenMP optional: compiles and runs deterministically even without OpenMP
 */

#include "rng_factory.hpp"
#include "rng_global.hpp"

#ifdef _OPENMP
  #include <omp.h>
#endif

#include <iterator> // std::begin/std::end

namespace mc {

namespace {

// splitmix64 mixer (Vigna), used here as a high-quality mixing function
inline std::uint64_t splitmix64_mix(std::uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

inline std::uint64_t combine_seeds(std::uint64_t base_seed,
                                   std::uint64_t stream_id,
                                   std::uint64_t thread_id)
{
    constexpr std::uint64_t GOLDEN = 0x9e3779b97f4a7c15ULL;

    std::uint64_t combined = base_seed;
    combined ^= splitmix64_mix(stream_id + GOLDEN);
    combined ^= splitmix64_mix((thread_id << 1) + 1ULL);

    return splitmix64_mix(combined);
}

/**
 * @brief Create mt19937 from a 64-bit seed
 *
 * Expands the 64-bit mixed seed into 8x 32-bit words deterministically.
 * This feeds std::seed_seq with more material than just {lo, hi}.
 */
inline std::mt19937 create_mt19937(std::uint64_t seed) {
    std::uint32_t data[8];

    std::uint64_t x = seed;
    for (int i = 0; i < 8; ++i) {
        x = splitmix64_mix(x + 0x9e3779b97f4a7c15ULL);
        data[i] = static_cast<std::uint32_t>(x & 0xFFFFFFFFULL);
    }

    std::seed_seq seq(std::begin(data), std::end(data));
    return std::mt19937(seq);
}

} // anonymous namespace

std::mt19937 make_engine(std::uint64_t stream_id) {
    const std::uint64_t base = static_cast<std::uint64_t>(get_global_seed());
    const std::uint64_t mixed = combine_seeds(base, stream_id, 0ULL);
    return create_mt19937(mixed);
}

std::mt19937 make_thread_engine(std::uint64_t stream_id) {
    const std::uint64_t base = static_cast<std::uint64_t>(get_global_seed());

    std::uint64_t tid = 0ULL;
#ifdef _OPENMP
    tid = static_cast<std::uint64_t>(omp_get_thread_num());
#endif

    const std::uint64_t mixed = combine_seeds(base, stream_id, tid);
    return create_mt19937(mixed);
}

std::mt19937 make_engine_with_seed(std::optional<std::uint32_t> base_seed,
                                   std::uint64_t stream_id)
{
    const std::uint64_t base = base_seed.has_value()
        ? static_cast<std::uint64_t>(*base_seed)
        : static_cast<std::uint64_t>(get_global_seed());

    const std::uint64_t mixed = combine_seeds(base, stream_id, 0ULL);
    return create_mt19937(mixed);
}

} // namespace mc