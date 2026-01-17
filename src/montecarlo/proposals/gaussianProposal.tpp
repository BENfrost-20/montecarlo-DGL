// gaussianProposal.tpp
#ifndef MONTECARLO_1_GAUSSIAN_PROPOSAL_TPP
#define MONTECARLO_1_GAUSSIAN_PROPOSAL_TPP

#include <stdexcept>
#include <cmath>     // std::log, std::exp, std::sqrt
#include <limits>    // std::numeric_limits

template <size_t dim>
void GaussianProposal<dim>::init_from_mu_sig_()
{
    // Validate sizes to avoid out-of-bounds / UB.
    if (mu.size() != dim || sig.size() != dim) {
        throw std::invalid_argument("GaussianProposal: mean and sigma must have size = dim.");
    }

    inv_sig2.resize(dim);

    double sum_log_inv_sigma = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        if (sig[i] <= 0.0) {
            throw std::invalid_argument("GaussianProposal: sigma must be > 0 for every dimension.");
        }

        inv_sig2[i] = 1.0 / (sig[i] * sig[i]);
        ndist[i] = std::normal_distribution<double>(mu[i], sig[i]);
        sum_log_inv_sigma += std::log(1.0 / sig[i]);
    }

    // log normalization constant for diagonal Gaussian:
    // phi(x) = (2pi)^(-d/2) * prod_i (1/sigma_i) * exp(-0.5 * sum_i ((x_i-mu_i)^2/sigma_i^2))
    const double log_2pi = std::log(2.0 * M_PI);
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
template <class Func>
GaussianProposal<dim>::GaussianProposal(const IntegrationDomain<dim>& d,
                                        Func&& f,
                                        std::mt19937& rng,
                                        std::size_t n_pilot,
                                        const std::vector<double>& init_mean,
                                        const std::vector<double>& init_sigma,
                                        double eps_sigma)
    : domain(d)
{
    mu.assign(dim, 0.0);
    sig.assign(dim, 1.0);

    estimate_moments_from_function_(std::forward<Func>(f), rng, n_pilot, init_mean, init_sigma, eps_sigma);
    init_from_mu_sig_();
}

template <size_t dim>
template <class Func>
void GaussianProposal<dim>::estimate_moments_from_function_(Func&& f,
                                                           std::mt19937& rng,
                                                           std::size_t n_pilot,
                                                           const std::vector<double>& init_mean,
                                                           const std::vector<double>& init_sigma,
                                                           double eps_sigma)
{
    if (n_pilot == 0) {
        throw std::invalid_argument("GaussianProposal: n_pilot must be > 0.");
    }
    if (init_mean.size() != dim || init_sigma.size() != dim) {
        throw std::invalid_argument("GaussianProposal: init_mean and init_sigma must have size = dim.");
    }
    if (eps_sigma <= 0.0) {
        throw std::invalid_argument("GaussianProposal: eps_sigma must be > 0.");
    }

    // Pilot distributions (independent from ndist).
    std::array<std::normal_distribution<double>, dim> pilot_dist{};
    for (size_t i = 0; i < dim; ++i) {
        if (init_sigma[i] <= 0.0) {
            throw std::invalid_argument("GaussianProposal: init_sigma must be > 0 for every dimension.");
        }
        pilot_dist[i] = std::normal_distribution<double>(init_mean[i], init_sigma[i]);
    }

    // Self-normalized weighted moments for target proportional to |f(x)|.
    std::vector<double> m1(dim, 0.0);
    std::vector<double> m2(dim, 0.0);
    double wsum = 0.0;

    // Cap attempts per pilot sample to avoid pathological infinite loops.
    const std::size_t max_tries = 10000;

    for (std::size_t k = 0; k < n_pilot; ++k) {
        geom::Point<dim> x;
        std::size_t tries = 0;

        // Generate a pilot point inside the domain using rejection.
        do {
            for (size_t i = 0; i < dim; ++i) {
                x[i] = pilot_dist[i](rng);
            }
            ++tries;
            if (tries >= max_tries) {
                break;
            }
        } while (!domain.isInside(x));

        if (tries >= max_tries) {
            // Skip this pilot sample if we couldn't land inside the domain.
            continue;
        }

        double w = static_cast<double>(f(x));
        w = std::abs(w);

        if (!std::isfinite(w) || w <= 0.0) {
            continue;
        }

        wsum += w;
        for (size_t i = 0; i < dim; ++i) {
            m1[i] += w * x[i];
            m2[i] += w * x[i] * x[i];
        }
    }

    // Fallback: if no effective weight, keep the initial pilot Gaussian.
    if (!(wsum > 0.0) || !std::isfinite(wsum)) {
        mu = init_mean;
        sig = init_sigma;
        return;
    }

    mu.resize(dim);
    sig.resize(dim);

    for (size_t i = 0; i < dim; ++i) {
        const double mean_i = m1[i] / wsum;
        const double ex2_i  = m2[i] / wsum;
        double var_i = ex2_i - mean_i * mean_i;

        if (!std::isfinite(var_i) || var_i < 0.0) {
            var_i = 0.0;
        }

        mu[i]  = mean_i;
        sig[i] = std::sqrt(var_i + eps_sigma * eps_sigma);
    }
}

template <size_t dim>
geom::Point<dim> GaussianProposal<dim>::sample(std::mt19937& rng) const
{
    geom::Point<dim> x;

    // Rejection sampling to enforce domain constraint.
    do {
        for (size_t i = 0; i < dim; ++i) {
            x[i] = ndist[i](rng);
        }
    } while (!domain.isInside(x));

    return x;
}

template <size_t dim>
double GaussianProposal<dim>::pdf(const geom::Point<dim>& x) const
{
    // Outside the domain => 0 (indicator constraint).
    if (!domain.isInside(x)) {
        return 0.0;
    }

    // Compute log(phi(x)) for numerical stability.
    double quad = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        const double diff = x[i] - mu[i];
        quad += diff * diff * inv_sig2[i];
    }

    return std::exp(log_norm_const - 0.5 * quad);
}

#endif // MONTECARLO_1_GAUSSIAN_PROPOSAL_TPP