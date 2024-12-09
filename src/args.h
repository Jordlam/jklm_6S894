#ifndef ARGS
#define ARGS

#include "include/tgaimage.h"
#include "include/obj_loader.h"
#include "include/fbx_loader.h"

#include <stdint.h>

struct Args {
    int32_t width;
    int32_t height;

    // To make: coordinates[vertices[0]] = {n0.x,n0.y,n0.z}
    int32_t n_triangle;
    // Stored in order like so: [n0.v1,n0.v2,n0.v3, n1.v1,n1.v2,n1.v3, ...]
    int32_t *vertices;
    // Stored like so: [v1.x, v5.x, v0.x, ...]
    float *vertex_x;
    // Stored like so: [v1.y, v5.y, v0.y, ...]
    float *vertex_y;
    // Stored like so: [v1.z, v5.z, v0.z, ...]
    float *vertex_z;

    // For imaging
    float *triangle_red;
    float *triangle_green;
    float *triangle_blue;
    float *triangle_alpha;
    TGAImage img;

    // TODO Right now only for simple model
    ufbx_vec2 cam_size;
    ufbx_vec3 cam_pos;
    ufbx_vec3 light_dir;
    // Maybe you want to do more so here they are
    Model model;
    Scene scene;

    // Methods
    void get_args(
        const std::string &obj_file,
        const std::string &mtl_path,
        const std::string &fbx_file
    );
    void clean_args();
};

#endif