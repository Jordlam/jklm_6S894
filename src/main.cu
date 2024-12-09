#include <cuda_runtime.h>
#include <iostream>
#include "cpu_rasterizer.h"

int main(int argc, char const *const *argv) {
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
    return 0;
}