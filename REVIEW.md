# Montecarlo-DGL code review
# My comments
Good presentation. I liked the fact that you tried to find practical applications.

#CODEX comments
## Overview
- The project implements Monte Carlo integration over multiple domains (hypersphere, hyperrectangle, hypercylinder, polytope) with classic MC, importance sampling, and Metropolis–Hastings MCMC.
- It includes proposal distributions (uniform, Gaussian, mixture), estimators (mean/volume), RNG utilities, and optimizers (PSO, GA).
- Apps provide benchmarks, a drone optimization case, a wind farm simulator, and RNG reproducibility tests.

Key source areas:
- `src/montecarlo/domains/*` – geometric domains and containment/volume helpers.
- `src/montecarlo/integrators/*` – MC/IS/MH integrators.
- `src/montecarlo/estimators/*` – mean/volume estimators.
- `src/montecarlo/mcmc/*` – Metropolis–Hastings sampler.
- `src/montecarlo/proposals/*` – uniform/gaussian/mixture proposals.
- `src/montecarlo/optimizers/*` – PSO and GA.
- `src/apps/*` – CLI and benchmarks.

## Possible compilation / build issues
1) **CMake OpenMP target used before it’s defined**
   - In `CMakeLists.txt`, `target_link_libraries(montecarlo_optimizers PRIVATE OpenMP::OpenMP_CXX)` appears **before** `find_package(OpenMP REQUIRED)`. On many CMake versions this is a configure-time error because the imported target doesn’t exist yet. Move the `find_package(OpenMP REQUIRED)` call before any use of `OpenMP::OpenMP_CXX`.

2) **README build steps miss the `cd build` step**
   - README says “Create the build directory and enter it,” but only runs `mkdir build` and then `cmake ..`. This will fail unless the user manually `cd build`. Consider adding that line explicitly.

3) **Non‑portable `M_PI` usage**
   - Files like `src/apps/benchmarks/integration_benchmarks.cpp` use `M_PI` without defining it. This can fail on MSVC unless `_USE_MATH_DEFINES` is set. Since C++20 is enabled, prefer `std::numbers::pi` from `<numbers>` for portability.

## Confirmed programming flaw
1) **MH integrator under‑samples when OpenMP uses fewer threads than `omp_get_max_threads()`**
   - In `src/montecarlo/integrators/MHintegrator.tpp`, the work partition uses `omp_get_max_threads()` outside the parallel region. If the runtime spawns fewer threads (e.g., `OMP_NUM_THREADS` is set), the total number of MH samples becomes **less than requested**.
   - I added an inline `@comment` in `src/montecarlo/integrators/MHintegrator.tpp` at the partitioning block to flag this.

## Improvement suggestions
- Avoid `add_compile_options(-w)` in `CMakeLists.txt`; it hides warnings that often catch bugs. Prefer `-Wall -Wextra -Wpedantic` (or equivalents) and fix warnings.
- In `MetropolisHastingsSampler`, consider optional domain checks or a policy for rejecting proposals outside the domain instead of relying on the target density to return 0. This prevents silent failures when user-supplied `p(x)` doesn’t follow that convention.
- In `Hypersphere::isInside`, compare squared norm to `radius*radius` to avoid `sqrt` per sample for a small speedup.
- In `GaussianProposal::sample`, creating a new `std::normal_distribution` each iteration is a bit costly. Consider caching distributions per dimension (similar to `UniformProposal`).
- The `proposal` parameter in `MHMontecarloIntegrator::integrate` is unused; consider removing it (if API permits) or explicitly marking it unused to silence warnings.
- `README.md` documents `montecarlo_1` and `drone_optimization` but not `wind_farm_simulator` and `test_rng_reproducibility`; add a short note so users discover them.

## Files touched
- `src/montecarlo/integrators/MHintegrator.tpp` – added an `@comment` note about OpenMP thread-count mismatch.
