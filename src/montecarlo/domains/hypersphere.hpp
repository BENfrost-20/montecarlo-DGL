#ifndef HYPERSPHERE_HPP
#define HYPERSPHERE_HPP

#include "integration_domain.hpp"
#include <utility>
#include "../geometry.hpp"

using namespace std;
using namespace geom;

/**
 * @brief N-dimensional ball (solid sphere)
 * @tparam dim Dimensionality
 * 
 * Represents a ball of specified radius centered at origin:
 * B = {x ∈ ℝⁿ : ||x|| ≤ r}
 * 
 * Volume computation uses the closed-form formula:
 * V = (π^(n/2) / Γ(n/2 + 1)) * r^n
 */
template <size_t dim>

class Hypersphere : public IntegrationDomain<dim>{
public:
    Hypersphere(double rad);

    Bounds<dim> getBounds() const override;

    double getBoxVolume() const override;

    bool isInside(const Point<dim> &point) const override;
private:
    double radius;
};

#include "hypersphere.tpp"

#endif //HYPERSPHERE_HPP