//
// Created by Giacomo Merlo on 14/01/26.
//
#include <omp.h>
#include "../RngManager.hpp"

template <std::size_t dim>
MeanEstimate<dim> MCMeanEstimator<dim>::estimate(const IntegrationDomain<dim>& domain,
             std::uint32_t seed,
             std::size_t n_samples,
             const std::function<double(const Point<dim>&)>& f) const
{
    if (n_samples <= 0) throw std::invalid_argument("n_samples must be > 0");

    auto bounds = domain.getBounds();
    for (size_t i = 0; i < dim; ++i) {
        dist[i] = std::uniform_real_distribution<double>(
            bounds[i].first, bounds[i].second
        );
    }

    const int T = omp_get_max_threads();

    const size_t base = n_samples / T;
    const size_t rem  = n_samples % T;

    double sum = 0.0;
    double sum2 = 0.0;

    std::size_t inside_total = 0;
    RngManager rngs(seed);



#pragma omp parallel for reduction(+:sum,sum2,inside_total)
    for (int tid = 0; tid < T; ++tid){

        // All threads get base samples, except the first rem threads which get one more
        const int n_local = base + (tid < rem ? 1 : 0);
        // One RNG per thread
        auto rng = rngs.make_rng(tid);

        for (int i=0; i< n_local; ++i) {
            geom::Point<dim> p;
            for (size_t i = 0; i < dim; ++i) {
                p[i] = dist[i](rng);
            }
            if (domain.isInside(p)) {
                    double term = f(p);
                    sum += term;
                    sum2 += term * term;
                    inside_total += 1;

            }
        }
    }

    MeanEstimate<dim> out;
    out.n_samples = n_samples;
    out.n_inside  = inside_total;
    out.mean = sum/n_samples;
    const  double e2 = sum2 / (n_samples);
    const  double var = std::max(0.0, e2 - out.mean * out.mean);
    out.stderr = (std::sqrt(var /(n_samples)));
    return out;
}