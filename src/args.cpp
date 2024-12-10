#include "args.h"
#include "cpu_rasterizer.h"

using Triangle2D = Triangle<Point2D>;
using Triangle3D = Triangle<Point3D>;

void Args::get_args(
    const std::string &obj_file,
    const std::string &mtl_path,
    const std::string &fbx_file
) {
    model.loadModel(obj_file, mtl_path);
    scene.loadFBX(fbx_file);
    // Right now only for simple model
    cam_size = scene.get_camera(0).camera->resolution;
    cam_pos = scene.get_camera(0).local_transform.translation;
    light_dir = scene.get_light(0).light->local_direction;

    // Now get usual buffers
    width = (int32_t)cam_size.x;
    height = (int32_t)cam_size.y;

    // Allocate
    n_triangle = (int32_t)(model.nfaces(0) / 3);
    vertices = (int32_t*)malloc(sizeof(int32_t) * n_triangle * 3);
    vertex_x = (float*)malloc(sizeof(float) * n_triangle * 3);
    vertex_y = (float*)malloc(sizeof(float) * n_triangle * 3);
    vertex_z = (float*)malloc(sizeof(float) * n_triangle * 3);
    z_buffer = (float*)malloc(sizeof(float) * width * height);

    triangle_red = (float*)malloc(sizeof(float) * n_triangle);
    triangle_green = (float*)malloc(sizeof(float) * n_triangle);
    triangle_blue = (float*)malloc(sizeof(float) * n_triangle);
    triangle_alpha = (float*)malloc(sizeof(float) * n_triangle);
    img = TGAImage(width, height, TGAImage::RGB);

    // Get values
    tinyobj::attrib_t attrib = model.get_attrib();
    tinyobj::shape_t shape = model.get_shape(0);
    for (int t = 0; t < n_triangle; t++) {
        std::array<Point3D, 3> world_triangle_vertices;
        for (int v = 0; v < 3; v++) {
            int v_idx = shape.mesh.indices[3 * t + v].vertex_index;
            vertices[3 * t + v] = v_idx;
            float x = attrib.vertices[3 * v_idx + 0];
            float y = attrib.vertices[3 * v_idx + 1];
            float z = attrib.vertices[3 * v_idx + 2];
            vertex_x[v_idx] = x;
            vertex_y[v_idx] = y;
            vertex_z[v_idx] = z;
            world_triangle_vertices[v] = Point3D({
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(z)
            });
        }
        // All of this for light value
        Triangle3D world_triangle(
            world_triangle_vertices[0],
            world_triangle_vertices[1],
            world_triangle_vertices[2]
        );
        Point3D norm = world_triangle.normal();
        Point3D light_direction({(float)light_dir.x, (float)light_dir.y, (float)light_dir.z});
        float light_intensity = norm.dot(light_direction);
        triangle_red[t] = 255 * light_intensity;
        triangle_green[t] = 255 * light_intensity;
        triangle_blue[t] = 255 * light_intensity;
        triangle_alpha[t] = (float)255;
    }
    return;
}

void Args::clean_args() {
    free((void*)vertices);
    free((void*)vertex_x);
    free((void*)vertex_y);
    free((void*)vertex_z);
    free((void*)z_buffer);

    free((void*)triangle_red);
    free((void*)triangle_green);
    free((void*)triangle_blue);
    free((void*)triangle_alpha);
    model.cleanup();
    scene.cleanup();
}
