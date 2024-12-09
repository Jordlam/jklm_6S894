#ifndef CPU_RASTERIZER
#define CPU_RASTERIZER

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>

struct Point2D {
    int x;
    int y;

    Point2D operator+(Point2D p);
    Point2D operator-(Point2D p);
    int dot(Point2D p);
};

struct Point3D {
    float x;
    float y;
    float z;
    // Point3D() : x(0.0f), y(0.0f), z(0.0f) {}  // QUESTION: Why does this cause errors?

    Point3D operator+(Point3D p);
    Point3D operator-(Point3D p);
    Point3D cross(Point3D p);
    float dot(Point3D p);
    float magnitude();
    Point3D normalize();
};

template<typename Point>
struct Triangle {
    Point a;
    Point b;
    Point c;

    Triangle<Point>(
        Point _a,
        Point _b,
        Point _c
    ) :
        a(_a), b(_b), c(_c)
    {}

    Point normal();
};

// Main render
int cpu_render();

#endif
