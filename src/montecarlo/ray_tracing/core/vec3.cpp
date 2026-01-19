#include "vec3.hpp"
#include <cmath>

// --- Indexing ---
float& vec3::operator[](int i) {
    return (i == 0) ? x : ((i == 1) ? y : z);
}

const float& vec3::operator[](int i) const {
    return (i == 0) ? x : ((i == 1) ? y : z);
}

// --- Arithmetic operators ---
vec3 vec3::operator*(float v) const {
    return vec3{x * v, y * v, z * v};
}

float vec3::operator*(const vec3& v) const {
    return x * v.x + y * v.y + z * v.z;
}

vec3 vec3::operator+(const vec3& v) const {
    return vec3{x + v.x, y + v.y, z + v.z};
}

vec3 vec3::operator-(const vec3& v) const {
    return vec3{x - v.x, y - v.y, z - v.z};
}

vec3 vec3::operator-() const {
    return vec3{-x, -y, -z};
}

vec3& vec3::operator*=(float v) {
    x *= v; y *= v; z *= v;
    return *this;
}

vec3& vec3::operator+=(const vec3& v) {
    x += v.x; y += v.y; z += v.z;
    return *this;
}

vec3& vec3::operator-=(const vec3& v) {
    x -= v.x; y -= v.y; z -= v.z;
    return *this;
}

vec3 operator*(float s, const vec3& v) {
    return vec3{v.x * s, v.y * s, v.z * s};
}

// --- Geometry helpers ---
float vec3::dot(const vec3& v) const {
    return x * v.x + y * v.y + z * v.z;
}

vec3 vec3::cross(const vec3& v) const {
    return vec3{
        y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x
    };
}

float vec3::norm2() const {
    return x * x + y * y + z * z;
}

float vec3::norm() const {
    return std::sqrt(norm2());
}

vec3 vec3::normalized() const {
    const float n2 = norm2();
    if (!(n2 > 0.f)) return vec3{};
    const float inv_n = 1.f / std::sqrt(n2);
    return (*this) * inv_n;
}

vec3& vec3::normalize() {
    const float n2 = norm2();
    if (!(n2 > 0.f)) {
        x = y = z = 0.f;
        return *this;
    }
    const float inv_n = 1.f / std::sqrt(n2);
    x *= inv_n; y *= inv_n; z *= inv_n;
    return *this;
}

bool vec3::is_finite() const {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

// --- Free functions ---
float dot(const vec3& a, const vec3& b) {
    return a.dot(b);
}

vec3 cross(const vec3& a, const vec3& b) {
    return a.cross(b);
}