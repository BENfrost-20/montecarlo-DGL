#ifndef RT_VEC3_HPP
#define RT_VEC3_HPP

#include <cmath>
#include <cstdint>

class vec3 {
public:
    float x, y, z;

    // --- Constructors ---
    constexpr vec3() : x(0.f), y(0.f), z(0.f) {}
    constexpr vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    // --- Indexing (compatible with old struct) ---
    float& operator[](int i);
    const float& operator[](int i) const;

    // --- Arithmetic operators (compatible) ---
    vec3  operator*(float v) const;
    float operator*(const vec3& v) const;   // dot product (legacy compatibility)
    vec3  operator+(const vec3& v) const;
    vec3  operator-(const vec3& v) const;
    vec3  operator-() const;

    vec3& operator*=(float v);
    vec3& operator+=(const vec3& v);
    vec3& operator-=(const vec3& v);

    friend vec3 operator*(float s, const vec3& v);

    // --- Geometry helpers ---
    float dot(const vec3& v) const;
    vec3  cross(const vec3& v) const;

    float norm2() const;
    float norm() const;

    vec3  normalized() const;
    vec3& normalize();

    bool is_finite() const;
};

// --- Free functions (explicit math style, overloads) ---
float dot(const vec3& a, const vec3& b);
vec3  cross(const vec3& a, const vec3& b);

#endif // RT_VEC3_HPP