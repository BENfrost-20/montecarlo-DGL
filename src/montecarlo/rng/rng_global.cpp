#include "rng_global.hpp"
#include <atomic>

namespace mc {

namespace {

    // Default seed (used if never explicitly initialized)
    std::atomic<std::uint32_t> g_global_seed{12345u};

    // Tracks whether the seed has been explicitly set
    std::atomic<bool> g_seed_initialized{false};

}

bool set_global_seed(std::uint32_t s) {
    bool expected = false;

    // Allow setting the seed ONLY if it was never initialized
    if (g_seed_initialized.compare_exchange_strong(
            expected,
            true,
            std::memory_order_acq_rel))
    {
        g_global_seed.store(s, std::memory_order_relaxed);
        return true;
    }

    return false;
}

std::uint32_t get_global_seed() {
    return g_global_seed.load(std::memory_order_relaxed);
}

bool is_global_seed_initialized() {
    return g_seed_initialized.load(std::memory_order_acquire);
}

} // namespace mc