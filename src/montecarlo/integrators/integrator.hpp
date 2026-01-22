/**
 * @file integrator.hpp
 * @brief Abstract base class for numerical integration in N-dimensional spaces.
 * @details Provides infrastructure for random number generation and
 * domain-based sampling initialization. Serves as the parent for concrete
 * integration strategies (uniform Monte Carlo, importance sampling, MCMC).
 */

#ifndef MONTECARLO_1_INTEGRATOR_HPP
#define MONTECARLO_1_INTEGRATOR_HPP

#include <random>
#include <functional>
#include <array>
#include <fstream>
#include <vector>

#include "../domains/integration_domain.hpp"
#include "../domains/hypersphere.hpp"
#include "../domains/hypercylinder.hpp"
#include "../domains/hyperrectangle.hpp"

namespace mc::integrators {

/**
 * @class Integrator
 * @brief Abstract base class for Monte Carlo integration in N dimensions.
 * @tparam dim The dimensionality of the integration domain.
 * @details Manages random number generation and provides utilities for
 * initializing sample points within the integration domain. Subclasses
 * implement specific integration algorithms (uniform sampling, importance sampling, MCMC).
 */
template <size_t dim>
class Integrator
{
protected:
    const mc::domains::IntegrationDomain<dim> &domain; ///< Reference to the integration domain.
    std::vector<std::mt19937> randomizer; ///< Per-thread random number generators.

    /**
     * @brief Initializes random samples uniformly distributed in the domain.
     * @param numbers The number of sample points to generate.
     * @return Vector of randomly sampled points within the domain bounds.
     * @details Generates points uniformly in the bounding box, writes to file
     * for visualization (hsphere_samples.dat, cylinder_samples.dat, etc.),
     * and returns as vector of Point<dim>.
     * @note This helper was used in earlier development; modern code uses RngManager.
     */
    std::vector<mc::geom::Point<dim>> initializeRandomizer(int numbers)
    {
        // Initialize seed sequence for random number generators
        std::seed_seq seq{1, 2, 3, 4, 5};
        std::vector<std::uint32_t> seeds(dim);
        seq.generate(seeds.begin(), seeds.end());

        // Create 'dim' independent random engines (one per dimension)
        std::array<std::mt19937, dim> engines;
        for (size_t i = 0; i < dim; ++i)
            engines[i].seed(seeds[i]);

        // Create uniform distributions for each dimension
        std::array<std::uniform_real_distribution<double>, dim> distributions;
        for (size_t i = 0; i < dim; ++i)
        {
            auto bounds = this->domain.getBounds();
            distributions[i] = std::uniform_real_distribution<double>(bounds[i].first,
                                                                 bounds[i].second);
        }

        // Reserve storage for samples
        std::vector<mc::geom::Point<dim>> random_numbers;
        random_numbers.reserve(numbers);

        // Open output file based on domain type for visualization
        std::ofstream outfile;
        if (typeid(domain) == typeid(mc::domains::Hypersphere<dim>))
        {
            outfile.open("hsphere_samples.dat");
        }
        else if (typeid(domain) == typeid(mc::domains::HyperCylinder<dim>))
        {
            outfile.open("cylinder_samples.dat");
        }
        else if (typeid(domain) == typeid(mc::domains::HyperRectangle<dim>))
        {
            outfile.open("hrectangle_samples.dat");
        }
        else
        {
            outfile.open("generic_samples.dat");
        }

        // Generate 'numbers' sample points, one per line in output file
        for (int j = 0; j < numbers; ++j)
        {
            mc::geom::Point<dim> x;
            for (size_t i = 0; i < dim; ++i)
            {
                x[i] = distributions[i](engines[i]);
                outfile << x[i];
                if (i + 1 < dim)
                    outfile << ' ';
            }
            random_numbers.push_back(x);
            outfile << "\n";
        }

        return random_numbers;
    }

public:
    /**
     * @brief Constructs an integrator for a specific domain.
     * @param d Reference to the integration domain.
     */
    explicit Integrator(const mc::domains::IntegrationDomain<dim> &d) : domain(d) {}

    virtual double integrate(const std::function<double(const mc::geom::Point<dim>&)>& f,
                             int n_samples,
                             const mc::proposals::Proposal<dim>& proposal,
                             std::uint32_t seed) = 0;

    /**
     * @brief Virtual destructor for proper polymorphic cleanup.
     */
    virtual ~Integrator() = default;
};

} // namespace mc::integrators

#endif // MONTECARLO_1_INTEGRATOR_HPP