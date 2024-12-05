#include <algorithm>
#include <cmath>
#include <cstdlib>
// #include <cuda_runtime.h>
#include <iostream>
#include <limits>

#include "model.h"
#include "tgaimage.h"

struct Point2D {
    int x;
    int y;

    Point2D operator+(Point2D p) {
        return {x + p.x, y + p.y};
    }

    Point2D operator-(Point2D p) {
        return {x - p.x, y - p.y};
    }

    int dot(Point2D p) {
        return x * p.x + y * p.y;
    }
};

struct Point3D {
    float x;
    float y;
    float z;

    // Point3D() : x(0.0f), y(0.0f), z(0.0f) {}  // QUESTION: Why does this cause errors?

    Point3D operator+(Point3D p) {
        return {x + p.x, y + p.y, z + p.z};
    }

    Point3D operator-(Point3D p) {
        return {x - p.x, y - p.y, z - p.z};
    }

    Point3D cross(Point3D p) {
        return {y * p.z - z * p.y, z * p.x - x * p.z, x * p.y  - y  * p.x};
    }

    float dot(Point3D p) {
        return x * p.x + y * p.y + z * p.z;
    }
    
    float magnitude() {
        return sqrt(x * x + y * y + z * z);
    }

    Point3D normalize() {
        float magnitude_reciprocal = 1 / magnitude();
        
        return {
            x * magnitude_reciprocal,
            y * magnitude_reciprocal, 
            z * magnitude_reciprocal
        };
    }
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

using Triangle2D = Triangle<Point2D>;
using Triangle3D = Triangle<Point3D>;

// `normal` only makes sense for `Triangle3D`.
template<>
Point3D Triangle3D::normal() {

    // Must be in this order (i.e., using `b` as our point of reference),
    // presumably because the cross product is not commutative, and the
    // other orders cause issues.
    Point3D u = a - b;
    Point3D v = c - b;
    
    // Point3D u = a - c;
    // Point3D v = b - c;

    return u.cross(v).normalize();
}

// Returns bounding box of `triangle`, with exclusive ends.
std::array<int, 4> bounding_box(
    Triangle2D triangle,
    int image_width,
    int image_height
) {
    auto [lowest_x, highest_x] = std::minmax({triangle.a.x, triangle.b.x, triangle.c.x});
    auto [lowest_y, highest_y] = std::minmax({triangle.a.y, triangle.b.y, triangle.c.y});

    // Bound by the image size.
    lowest_x = std::max(lowest_x - 1, 0);
    lowest_y = std::max(lowest_y - 1, 0);
    highest_x = std::min(highest_x + 1, image_width - 1);
    highest_y = std::min(highest_y + 1, image_height - 1);

    return {lowest_x, highest_x, lowest_y + 1, highest_y + 1};
}

bool point_inside_triangle(Triangle2D triangle, Point2D point) {
    // Calculate barycentric coordinates using code adapted
    // from this page: https://blackpawn.com/texts/pointinpoly/

    Point2D v0 = triangle.c - triangle.a;
    Point2D v1 = triangle.b - triangle.a;
    Point2D v2 = point - triangle.a;

    int dot00 = v0.dot(v0);
    int dot01 = v0.dot(v1);
    int dot02 = v0.dot(v2);
    int dot11 = v1.dot(v1);
    int dot12 = v1.dot(v2);

    double invDenom = 1.0 / static_cast<double>(dot00  * dot11 - dot01  * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    // Determine whether the point is in the triangle.
    return (u >= 0) && (v >= 0) && (u + v < 1);
}

template<int IMAGE_WIDTH = 2000, int IMAGE_HEIGHT = 2000>
void draw_triangle(
    Triangle2D triangle,
    TGAColor color,
    TGAImage &image,
    std::vector<int> max_pixel_z
) {
    // Exclusive ends.
    auto [start_x, end_x, start_y, end_y] = bounding_box(triangle, IMAGE_WIDTH, IMAGE_HEIGHT);
    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            Point2D point = {x, y};
            if (point_inside_triangle(triangle, point)) {
                image.set(x, y, color); 
            }
        }
    }
}

int main(int argc, char** argv) {
    constexpr int FRAME_WIDTH = 2500;
    constexpr int FRAME_HEIGHT = 2500;
    std::vector<int> max_pixel_z(FRAME_WIDTH * FRAME_HEIGHT, std::numeric_limits<int>::min());

    TGAImage frame(FRAME_WIDTH, FRAME_HEIGHT, TGAImage::RGB);

    Model model("head.obj");
    Point3D light_direction({0.0, 0.0, -1.0});

    for (int t = 0; t < model.nfaces(); ++t) {
        std::array<Point3D, 3> world_triangle_vertices;
        std::array<Point2D, 3> screen_triangle_vertices;
        for (int v = 0; v < 3; ++v) {
            auto [world_x, world_y, world_z] = model.vert(t, v);
            world_triangle_vertices[v] = Point3D({
                static_cast<float>(world_x),
                static_cast<float>(world_y),
                static_cast<float>(world_z)
            });

            int screen_x = (world_x + 1.0) * static_cast<double>(FRAME_WIDTH) / 2.0;
            int screen_y = (world_y + 1.0) * static_cast<double>(FRAME_HEIGHT) / 2.0;
            screen_triangle_vertices[v] = Point2D({screen_x, screen_y});
        }

        Triangle2D screen_triangle(
            screen_triangle_vertices[0],
            screen_triangle_vertices[1],
            screen_triangle_vertices[2]
        );

        Triangle3D world_triangle(
            world_triangle_vertices[0],
            world_triangle_vertices[1],
            world_triangle_vertices[2]
        );

        Point3D norm = world_triangle.normal();
        float light_intensity = norm.dot(light_direction);
        if (light_intensity > 0) {
            TGAColor color(
                255 * light_intensity,
                255 * light_intensity,
                255 * light_intensity,
                255
            );

            draw_triangle<FRAME_WIDTH, FRAME_HEIGHT>(screen_triangle, color, frame, max_pixel_z);
        }
    }

    // Triangle triangle({10, 10}, {100, 30}, {190, 160});
    // draw_triangle(triangle, frame, frame_width, frame_height);

    // Not sure why the line below is not necessary in my implementation, but whatever.
    // frame.flip_vertically(); // to place the origin in the bottom left corner of the image

    frame.write_tga_file("framebuffer.tga");

    return 0;
}