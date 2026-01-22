/**
 * @file gaussianProposal.tpp
 * @brief GaussianProposal template implementation.
 * @details Implements multivariate Gaussian sampling with PDF evaluation.
 * Samples are drawn from the full Gaussian (no domain truncation).
 * 
 * IMPORTANT:
 * - sample(rng) draws from the full Gaussian in R^dim (NO rejection).
 * - pdf(x) returns the corresponding full Gaussian density (NO domain indicator).
 *
 * Domain constraints (if any) must be handled by the estimator via:
 *   if(domain.isInside(p)) { ... }
 *
 * This guarantees sample() and pdf() are always coherent with the Proposal interface.
 */

// gaussianProposal.tpp
#ifndef MONTECARLO_1_GAUSSIAN_PROPOSAL_TPP
#define MONTECARLO_1_GAUSSIAN_PROPOSAL_TPP

#include <limits>   // std::numeric_limits
#include <utility>  // std::move
#include <numbers>

namespace mc {
namespace proposals {

/**
 * @brief Initialize normalization constant and precision from mean and standard deviation.
 * @tparam dim Dimensionality parameter.
 * 
 * @details Precomputes:
 * - inv_sig2[i] = 1 / (σᵢ²) for each dimension
 * - log_norm_const = log(det(Σ)^(-1/2) / (2π)^(n/2))
 *   = -n/2 * log(2π) + ∑ᵢ log(1/σᵢ)
 * 
 * This avoids recomputation during PDF evaluations.
 * 
 * @throws std::invalid_argument If dimensions mismatch or σᵢ ≤ 0.
 */
template <size_t dim>
void GaussianProposal<dim>::init_from_mu_sig_()
{
    if (mu.size() != dim || sig.size() != dim)
        throw std::invalid_argument("GaussianProposal: mean and sigma must have size = dim.");

    inv_sig2.resize(dim);

    double sum_log_inv_sigma = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        if (!(sig[i] > 0.0) || !std::isfinite(sig[i]))
            throw std::invalid_argument("GaussianProposal: sigma must be finite and > 0 for every dimension.");

        inv_sig2[i] = 1.0 / (sig[i] * sig[i]);

        sum_log_inv_sigma += std::log(1.0 / sig[i]);
    }

    const double pi = std::numbers::pi;
    const double log_2pi = std::log(2.0 * pi);
    log_norm_const = -0.5 * static_cast<double>(dim) * log_2pi + sum_log_inv_sigma;
}

/**
 * @brief Construct a multivariate Gaussian proposal distribution.
 * @tparam dim Dimensionality parameter.
 * @param d Integration domain (stored for consistency, may be used later).
 * @param mean Vector of means (μ) for each dimension. Size must equal dim.
 * @param sigma Vector of standard deviations (σ) for each dimension. Size must equal dim.
 *              All σᵢ must be > 0 and finite.
 * 
 * @throws std::invalid_argument If vector sizes don't match dim or σᵢ is invalid.
 * 
 * @details Initializes a diagonal Gaussian with independent components:
 * q(x) = ∏ᵢ N(xᵢ; μᵢ, σᵢ²)
 */
template <size_t dim>
GaussianProposal<dim>::GaussianProposal(const mc::domains::IntegrationDomain<dim>& d,
                                        const std::vector<double>& mean,
                                        const std::vector<double>& sigma)
    : domain(d), mu(mean), sig(sigma)
{
    init_from_mu_sig_();
}

/**
 * @brief Draw a random sample from the multivariate Gaussian.
 * @tparam dim Dimensionality parameter.
 * @param rng Mersenne Twister random generator.
 * @return Point sampled from N(μ, Σ) where Σ is diagonal with σᵢ² on diagonal.
 * 
 * @details Generates independent normal samples for each dimension:
 * xᵢ ~ N(μᵢ, σᵢ²). Note: samples are drawn from the FULL Gaussian,
 * not truncated to any domain. Domain handling is the caller's responsibility.
 */
template <size_t dim>
mc::geom::Point<dim> GaussianProposal<dim>::sample(std::mt19937& rng) const
{
    mc::geom::Point<dim> x;
    for (size_t i = 0; i < dim; ++i){
        std::normal_distribution<double> d(mu[i], sig[i]);
        x[i] = d(rng);
    }
    return x;
}

/**
 * @brief Evaluate the Gaussian probability density function at a point.
 * @tparam dim Dimensionality parameter.
 * @param x Query point.
 * @return q(x) = exp(log_norm_const - 0.5 * ∑ᵢ ((xᵢ - μᵢ)² / σᵢ²))
 * 
 * @details Uses precomputed normalization constant and inverse variances
 * for efficiency. Returns 0.0 for extreme tails where log is non-finite.
 * 
 * Time complexity: O(dim).
 */
template <size_t dim>
double GaussianProposal<dim>::pdf(const mc::geom::Point<dim>& x) const
{
    // Full Gaussian density on R^dim (no domain indicator).
    double quad = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double diff = x[i] - mu[i];
        quad += diff * diff * inv_sig2[i];
    }

    // exp(log_norm_const - 0.5 * quad)
    const double logp = log_norm_const - 0.5 * quad;

    // Optional safety: avoid returning NaN for extreme tails
    if (!std::isfinite(logp))
        return 0.0;

    return std::exp(logp);
}

} // namespace proposals
} // namespace mc

#endif // MONTECARLO_1_GAUSSIAN_PROPOSAL_TP