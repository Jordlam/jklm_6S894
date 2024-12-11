// #include "cpu_rasterizer.h"

// Point2D Point2D::operator+(Point2D p) {
//     return {x + p.x, y + p.y};
// }
// Point2D Point2D::operator-(Point2D p) {
//     return {x - p.x, y - p.y};
// }
// int Point2D::dot(Point2D p) {
//     return x * p.x + y * p.y;
// }

// Point3D Point3D::operator+(Point3D p) {
//     return {x + p.x, y + p.y, z + p.z};
// }
// Point3D Point3D::operator-(Point3D p) {
//     return {x - p.x, y - p.y, z - p.z};
// }
// Point3D Point3D::cross(Point3D p) {
//     return {y * p.z - z * p.y, z * p.x - x * p.z, x * p.y  - y  * p.x};
// }
// float Point3D::dot(Point3D p) {
//     return x * p.x + y * p.y + z * p.z;
// }
// float Point3D::magnitude() {
//     return sqrt(x * x + y * y + z * z);
// }
// Point3D Point3D::normalize() {
//     float magnitude_reciprocal = 1 / magnitude();    
//     return {
//         x * magnitude_reciprocal,
//         y * magnitude_reciprocal, 
//         z * magnitude_reciprocal
//     };
// }

// using Triangle2D = Triangle<Point2D>;
// using Triangle3D = Triangle<Point3D>;

// // `normal` only makes sense for `Triangle3D`.
// template<>
// Point3D Triangle3D::normal() {
//     // Must be in this order (i.e., using `b` as our point of reference),
//     // presumably because the cross product is not commutative, and the
//     // other orders cause issues.
//     Point3D u = a - b;
//     Point3D v = c - b;
    
//     // Point3D u = a - c;
//     // Point3D v = b - c;

//     return u.cross(v).normalize();
// }

// // Returns bounding box of `triangle`, with exclusive ends.
// std::array<int, 4> bounding_box(
//     Triangle2D triangle,
//     int image_width,
//     int image_height
// ) {
//     auto [lowest_x, highest_x] = std::minmax({triangle.a.x, triangle.b.x, triangle.c.x});
//     auto [lowest_y, highest_y] = std::minmax({triangle.a.y, triangle.b.y, triangle.c.y});

//     // Bound by the image size.
//     lowest_x = std::max(lowest_x, 0);
//     lowest_y = std::max(lowest_y, 0);
//     highest_x = std::min(highest_x + 1, image_width);
//     highest_y = std::min(highest_y + 1, image_height);

//     return {lowest_x, highest_x, lowest_y, highest_y};
// }

// // Returns barycentric coordinates for vertex c and b, respectively.
// // Uses code adapted from this page: https://blackpawn.com/texts/pointinpoly/
// std::array<float, 2> barycentric_coordinates(Triangle2D triangle, Point2D point) {
//     Point2D v0 = triangle.c - triangle.a;
//     Point2D v1 = triangle.b - triangle.a;
//     Point2D v2 = point - triangle.a;

//     int dot00 = v0.dot(v0);
//     int dot01 = v0.dot(v1);
//     int dot02 = v0.dot(v2);
//     int dot11 = v1.dot(v1);
//     int dot12 = v1.dot(v2);

//     float invDenom = 1.0 / static_cast<float>(dot00  * dot11 - dot01  * dot01);
//     float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
//     float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

//     return {u, v};
// }

// bool point_inside_triangle(Triangle2D triangle, Point2D point) {
//     auto [u, v] = barycentric_coordinates(triangle, point);

//     // Determine whether the point is in the triangle.
//     return (u >= 0) && (v >= 0) && (u + v <= 1);
// }

// bool pixel_visible_update(
//     Triangle2D triangle,
//     Point2D point,
//     std::array<float, 3> triangle_z_coords,
//     std::vector<float> &max_pixel_z,
//     int image_width
// ) {
//     auto [bary_c, bary_b] = barycentric_coordinates(triangle, point);
//     std::array<decltype(bary_c), 3> barycentric_coord = {1 - bary_c - bary_b, bary_b, bary_c};

//     float pixel_z_value = 0;
//     for (int i = 0; i < 3; ++i) {
//         pixel_z_value += static_cast<float>(barycentric_coord[i] * triangle_z_coords[i]);
//     }

//     bool pixel_is_visible = pixel_z_value > max_pixel_z[point.y * image_width + point.x];
//     if (pixel_is_visible) {
//         max_pixel_z[point.y * image_width + point.x] = pixel_z_value;
//     }
    
//     return pixel_is_visible;
// }

// template<int IMAGE_WIDTH = 2000, int IMAGE_HEIGHT = 2000>
// void draw_triangle(
//     Triangle2D triangle,
//     TGAColor color,
//     TGAImage &image,
//     std::vector<float>& max_pixel_z,
//     std::array<float, 3> triangle_z_coords
// ) {
//     // Exclusive ends.
//     auto [start_x, end_x, start_y, end_y] = bounding_box(triangle, IMAGE_WIDTH, IMAGE_HEIGHT);
//     for (int y = start_y; y < end_y; ++y) {
//         for (int x = start_x; x < end_x; ++x) {
//             Point2D point = {x, y};

//             if (
//                 point_inside_triangle(triangle, point)
//                 && pixel_visible_update(
//                     triangle,
//                     point,
//                     triangle_z_coords,
//                     max_pixel_z,
//                     IMAGE_WIDTH)
//             ) {
//                 image.set(x, y, color);
//             } 
//         }
//     }
// }

// int cpu_render() {
//     constexpr int FRAME_WIDTH = 3500;
//     constexpr int FRAME_HEIGHT = FRAME_WIDTH;
//     TGAImage frame(FRAME_WIDTH, FRAME_HEIGHT, TGAImage::RGB);

//     // Parse objects and triangles
//     Model model;
//     model.loadModel("data/head.obj");
//     // model.loadModel("data/amongus_triangles.obj", "data/");

//     // Parse scene with cameras and lights
//     Scene scene;
//     scene.loadFBX("data/views.fbx");
//     Point3D light_direction({0.0, 0.0, -1.0});
//     std::vector<float> max_pixel_z(FRAME_WIDTH * FRAME_HEIGHT, std::numeric_limits<float>::lowest());

//     // TODO Given we only have one object in our scene: object i = 0
//     int i = 0;
//     for (int t = 0; t < model.nfaces(i); t+=3) {
//         std::array<Point3D, 3> world_triangle_vertices;
//         std::array<Point2D, 3> screen_triangle_vertices;
//         std::array<float, 3> z_coords;
//         for (int v = 0; v < 3; ++v) {
//             std::vector<float> vertices = model.vert(i, t, v);
//             float world_x = vertices[0];
//             float world_y = vertices[1];
//             float world_z = vertices[2];
//             world_triangle_vertices[v] = Point3D({
//                 static_cast<float>(world_x),
//                 static_cast<float>(world_y),
//                 static_cast<float>(world_z)
//             });

//             int screen_x = (world_x + 1.0) * static_cast<double>(FRAME_WIDTH) / 2.0;
//             int screen_y = (world_y + 1.0) * static_cast<double>(FRAME_HEIGHT) / 2.0;
//             screen_triangle_vertices[v] = Point2D({screen_x, screen_y});

//             z_coords[v] = static_cast<float>(world_z);
//         }

//         Triangle2D screen_triangle(
//             screen_triangle_vertices[0],
//             screen_triangle_vertices[1],
//             screen_triangle_vertices[2]
//         );

//         Triangle3D world_triangle(
//             world_triangle_vertices[0],
//             world_triangle_vertices[1],
//             world_triangle_vertices[2]
//         );

//         Point3D norm = world_triangle.normal();
//         float light_intensity = norm.dot(light_direction);
//         if (light_intensity > 0) {
//             TGAColor color(
//                 255 * light_intensity,
//                 255 * light_intensity,
//                 255 * light_intensity,
//                 255
//             );

//             draw_triangle<FRAME_WIDTH, FRAME_HEIGHT>(screen_triangle, color, frame, max_pixel_z, z_coords);
//         } 
//     }

//     // Triangle triangle({10, 10}, {100, 30}, {190, 160});
//     // draw_triangle(triangle, frame, frame_width, frame_height);

//     // Not sure why the line below is not necessary in my implementation, but whatever.
//     // frame.flip_vertically(); // to place the origin in the bottom left corner of the image

//     printf("Saving drawing...\n");
//     frame.write_tga_file("out/framebuffer.tga");

//     scene.cleanup();
//     return 0;
// }
