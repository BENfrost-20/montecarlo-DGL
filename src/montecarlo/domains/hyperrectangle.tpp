#include <cmath>
#include "../geometry.hpp"
#include "integration_domain.hpp"
#include <array>
#include <utility>

namespace mc::domains {

template <size_t dim>
HyperRectangle<dim>::HyperRectangle(std::array<double, dim> &dims):
    dimensions(dims)
{};


template <size_t dim>
auto HyperRectangle<dim>::getBounds() const -> mc::geom::Bounds<dim> {
    mc::geom::Bounds<dim> bounds;
    for(size_t i = 0; i < dim; ++i)
        bounds[i] = std::make_pair(-dimensions[i]/2, dimensions[i]/2);
    return bounds;
}

template <size_t dim>
double HyperRectangle<dim>::getBoxVolume() const{
    double volume = 1;
    for(size_t i = 0; i < dim; ++i) {
        volume = volume * dimensions[i];
    }
    return volume;
}

template <size_t dim>
bool HyperRectangle<dim>::isInside(const mc::geom::Point<dim> &point) const{
    bool inside = true;
        for(size_t i = 0; i < dim; ++i) {
            if (point[i] < -dimensions[i]/2 || point[i] > dimensions[i]/2) {
                inside = false;
            }
        }
    return inside;
}

} // namespace mc::domains