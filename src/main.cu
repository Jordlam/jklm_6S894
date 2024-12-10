#include <cuda_runtime.h>
#include <iostream>

#include "args.h"
#include "cpu_rasterizer.h"

int main(int argc, char const *const *argv) {
    // Get buffers
    std::string obj_file = "data/head.obj";
    std::string mtl_path = "data/";
    std::string fbx_file = "data/views.fbx";
    Args args;
    args.get_args(obj_file.c_str(), mtl_path.c_str(), fbx_file.c_str());

    // Just for testing...
    printf("\nTesting if values propagate to CUDA...\n");
    printf("Image width %i\n", args.width);
    printf("Image height %i\n", args.height);
    printf("Number of triangles %i\n", args.n_triangle);
    printf("Testing Triangle 0...\n");
    int idx = args.vertices[0];
    printf("Triangle 0 with vertex %i\n", idx);
    printf("Vertex %i with x at %f\n", idx, args.vertex_x[idx]);
    printf("Vertex %i with y at %f\n", idx, args.vertex_y[idx]);
    printf("Vertex %i with z at %f\n", idx, args.vertex_z[idx]);
    printf("Triangle 0 with red %f\n", args.triangle_red[0]);
    printf("Triangle 0 with green %f\n", args.triangle_green[0]);
    printf("Triangle 0 with blue %f\n", args.triangle_blue[0]);
    printf("Triangle 0 with alpha %f\n", args.triangle_alpha[0]);

    printf("Camera width %f\n", args.cam_size.x);
    printf("Camera height %f\n", args.cam_size.y);
    printf("Camera x %f\n", args.cam_pos.x);
    printf("Camera y %f\n", args.cam_pos.y);
    printf("Camera z %f\n", args.cam_pos.z);
    printf("Light direction {%f, %f, %f}\n", args.light_dir.x, args.light_dir.y, args.light_dir.z);
    std::vector<int> projected = args.scene.project(args.vertex_x[idx], args.vertex_y[idx], args.vertex_z[idx]);
    printf("We projected the vertex to: %i, %i, %i\n", projected[0], projected[1], projected[2]);
    printf("\nEnd of args text...\n");

    // CPU
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cpu_render();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time: %f ms\n", milliseconds);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // TODO GPU
    args.clean_args();
    return 0;
}