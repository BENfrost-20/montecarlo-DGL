/**
 * @file rng_global.hpp
 * @brief Global seed management for Monte Carlo library
 * @author Generated for montecarlo-DGL refactoring
 * @date 2026-01-21
 * 
 * Provides a centralized, thread-safe global seed that all library components
 * can use for reproducible random number generation.
 * 
 * USAGE:
 * @code
 *   // At program start, before any parallel or stochastic operations:
 *   mc::set_global_seed(12345u);
 *   
 *   // Later, retrieve the seed (for logging, debugging, etc.):
 *   std::uint32_t s = mc::get_global_seed();
 * @endcode
 * 
 * @note set_global_seed() should be called ONCE at startup, before launching
 *       OpenMP parallel regions or any Monte Carlo algorithms. Changing the
 *       seed mid-execution may lead to non-deterministic behavior.
 */

#ifndef MONTECARLO_RNG_GLOBAL_HPP
#define MONTECARLO_RNG_GLOBAL_HPP

#include <cstdint>

namespace mc {

/**
 * @brief Set the global seed used by all library RNG components
 * @param s The seed value to use
 * 
 * Thread-safe (atomic store). Should be called once at program startup.
 * Default value is 12345u if never called.
 */
void set_global_seed(std::uint32_t s);

/**
 * @brief Get the current global seed
 * @return The current global seed value
 * 
 * Thread-safe (atomic load).
 */
std::uint32_t get_global_seed();

} // namespace mc

#endif // MONTECARLO_RNG_GLOBAL_HPP
