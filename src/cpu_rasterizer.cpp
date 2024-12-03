#include <algorithm>
#include <cmath>
// #include <cuda_runtime.h>
#include <iostream>

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

    Point3D operator+(Point3D p) {
        return {x + p.x, y + p.y, z + p.z};
    }

    Point3D operator-(Point3D p) {
        return {x - p.x, y - p.y, z - p.z};
    }

    float dot(Point3D p) {
        return x * p.x + y * p.y + z * p.z;
    }
};

struct Triangle {
    Point2D a;
    Point2D b;
    Point2D c;

    Triangle(
        Point2D _a,
        Point2D _b,
        Point2D _c
    ) :
        a(_a), b(_b), c(_c)
    {}
};

// Returns bounding box of `triangle`, with exclusive ends.
std::array<int, 4> bounding_box(
    Triangle triangle,
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

bool point_inside_triangle(Triangle triangle, Point2D point) {
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

void draw_triangle(
    Triangle triangle,
    TGAImage &image,
    int image_width,
    int image_height
) {
    // Exclusive ends.
    auto [start_x, end_x, start_y, end_y] = bounding_box(triangle, image_width, image_height);
    static uint8_t rand = 0;
    rand ^= 1;
    for (int y = start_y; y < end_y; ++y) {
        for (int x = start_x; x < end_x; ++x) {
            Point2D point = {x, y};
            if (point_inside_triangle(triangle, point)) {
                TGAColor red = TGAColor(255, 0, 0, 255);
                TGAColor white = TGAColor(255, 255, 255, 255);
                image.set(x, y, rand ? red : white);
            }
        }
    }
}

int main(int argc, char** argv) {
    int frame_width = 200;
    int frame_height = 200;
    TGAImage frame(frame_width, frame_height, TGAImage::RGB);

    Model model("head.obj");

    for (int t = 0; t < model.nfaces(); ++t) {
        std::array<Point2D, 3> triangle_vertices;
        for (int v = 0; v < 3; ++v) {
            auto [x, y, _] = model.vert(t, v);
            int screen_x = std::floor((x + 1.0) * static_cast<double>(frame_width) / 2.0);
            int screen_y = std::floor((y + 1.0) * static_cast<double>(frame_height) / 2.0);
            triangle_vertices[v] = {screen_x, screen_y};
        }
        
        Triangle triangle(triangle_vertices[0], triangle_vertices[1], triangle_vertices[2]);
        draw_triangle(triangle, frame, frame_width, frame_height);
    }

    // Triangle triangle({10, 10}, {100, 30}, {190, 160});
    // draw_triangle(triangle, frame, frame_width, frame_height);

    // Not sure why the line below is not necessary in my implementation, but whatever.
    // frame.flip_vertically(); // to place the origin in the bottom left corner of the image

    frame.write_tga_file("framebuffer.tga");

    return 0;
}