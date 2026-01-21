/**
 * @file rng_global.cpp
 * @brief Implementation of global seed management
 * @author Generated for montecarlo-DGL refactoring
 * @date 2026-01-21
 * 
 * Uses std::atomic for thread-safe access to the global seed.
 * No mutex is needed since we only do simple load/store operations.
 */

#include "rng_global.hpp"
#include <atomic>

namespace mc {

namespace {
    /**
     * @brief Atomic storage for the global seed
     * 
     * Default value is 12345u, consistent with the library's original default.
     * Using std::memory_order_seq_cst for maximum safety.
     */
    std::atomic<std::uint32_t> g_global_seed{12345u};
}

void set_global_seed(std::uint32_t s) {
    g_global_seed.store(s, std::memory_order_seq_cst);
}

std::uint32_t get_global_seed() {
    return g_global_seed.load(std::memory_order_seq_cst);
}

} // namespace mc
