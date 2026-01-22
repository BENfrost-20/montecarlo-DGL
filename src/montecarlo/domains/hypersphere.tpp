#include <cmath>
#include "../geometry.hpp"
#include "integration_domain.hpp"

namespace mc::domains {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <size_t dim>
Hypersphere<dim>::Hypersphere(double rad):
    radius(rad)
{}


template <size_t dim>
auto Hypersphere<dim>::getBounds() const -> mc::geom::Bounds<dim> {
    mc::geom::Bounds<dim> bounds;
    for(size_t i = 0; i < dim; ++i)
        bounds[i] = std::make_pair(-radius, radius);
    return bounds;
}

template <size_t dim>
double Hypersphere<dim>::getBoxVolume() const{
    return std::pow(2 * radius, dim);
}
template <size_t dim>
bool Hypersphere<dim>::isInside(const mc::geom::Point<dim> &point) const{
    double norm = 0;
    for(size_t k = 0; k<dim; ++k){
        norm+=std::pow(point[k], 2);
    }
    return std::sqrt(norm) <= radius;
}

} // namespace mc::domains