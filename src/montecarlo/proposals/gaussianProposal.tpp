// gaussianProposal.tpp
#ifndef MONTECARLO_1_GAUSSIAN_PROPOSAL_TPP
#define MONTECARLO_1_GAUSSIAN_PROPOSAL_TPP

#include <limits>   // std::numeric_limits
#include <utility>  // std::move
#include <numbers>

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

template <size_t dim>
GaussianProposal<dim>::GaussianProposal(const IntegrationDomain<dim>& d,
                                        const std::vector<double>& mean,
                                        const std::vector<double>& sigma)
    : domain(d), mu(mean), sig(sigma)
{
    init_from_mu_sig_();
}

template <size_t dim>
geom::Point<dim> GaussianProposal<dim>::sample(std::mt19937& rng) const
{
    geom::Point<dim> x;
    for (size_t i = 0; i < dim; ++i){
        std::normal_distribution<double> d(mu[i], sig[i]);
        x[i] = d(rng);
    }
    return x;
}

template <size_t dim>
double GaussianProposal<dim>::pdf(const geom::Point<dim>& x) const
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

#endif // MONTECARLO_1_GAUSSIAN_PROPOSAL_TP