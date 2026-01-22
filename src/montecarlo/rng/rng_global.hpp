#ifndef MONTECARLO_RNG_GLOBAL_HPP
#define MONTECARLO_RNG_GLOBAL_HPP

#include <cstdint>

namespace mc {
namespace rng {


/**
 * @brief Set the global seed used by all library RNG components.
 *        This function can be called ONLY ONCE.
 *
 * @param s The seed value to use.
 * @return true if the seed was successfully set,
 *         false if the seed was already initialized.
 *
 * Thread-safe. If the seed has already been set, the call has no effect.
 */
bool set_global_seed(std::uint32_t s);

/**
 * @brief Get the current global seed.
 * @return The current global seed value.
 *
 * Thread-safe.
 */
std::uint32_t get_global_seed();

/**
 * @brief Check whether the global seed has been explicitly initialized.
 * @return true if set_global_seed() was successfully called.
 */
bool is_global_seed_initialized();

} //namespace rng
} // namespace mc

#endif // MONTECARLO_RNG_GLOBAL_HPP