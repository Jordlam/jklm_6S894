#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>







// #include <cuda_runtime.h>
// #include <iostream>
// #include "cpu_rasterizer.h"

// int main(int argc, char const *const *argv) {
//     // CPU
//     cudaEvent_t start, stop;
//     float milliseconds = 0;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);

//     cpu_render();

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Time: %f ms\n", milliseconds);
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     // TODO GPU
//     return 0;
// }

// __global__ void draw_triangles(
//     uint32_t max_triangles_per_tile
// ) {
//     uint32_t x = blockIdx.x * TILE_SIZE + threadIdx.x;
//     uint32_t y = blockIdx.y * TILE_SIZE + threadIdx.y

//     uint32_t tile_index = blockIdx.y * gridDim.x + blockIdx.x; 
//     uint32_t tile_triangles_start = tile_index * max_triangles_per_tile; 

//     int32_t pixel_idx = y * width + x;
//     float pixel_red = 1.0f;
//     float pixel_green = 1.0f;
//     float pixel_blue = 1.0f;

//     // for every triangle in the tile:
//     for (
//         int i = tile_triangles_start;
//         (i < tile_triangles_start + max_triangles_per_tile) && map[i] < n_triangles;
//         ++i
//     ) {

//         x_index = args.vertices[triangle_num][0]
//         y_index = args.vertices[triangle_num][1]
//         z_index = args.vertices[triangle_num][2]

//         x = args.vertex_x[x_index]
//         y = args.vertex_y[y_index]
//         z = args.vertex_z[z_index]

//         Point3D world_triangle_vertices[3];
//         Point2D screen_triangle_vertices[3];
//         float z_coords[3];
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

//         if (point_inside_triangle(triangle, pixel)) {
//             if (
//                 pixel_visible_update(
//                     triangle,
//                     pixel,
//                     triangle_z_coords,
//                     max_pixel_z,
//                     IMAGE_WIDTH)
//             ) {
//                 image.set(x, y, color)
//             }
//         }
//     }
// }







// New Additions.
#include <cuda/std/array>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};

// Timer code from a Piazza post.
struct Timer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float milliseconds;

    void start() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        CUDA_CHECK(cudaEventRecord(start_event));
    }

    void stop(const std::string& s) {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
        printf("%s: %0f ms\n", s.c_str(), milliseconds);

        CUDA_CHECK(cudaEventDestroy(start_event));
        CUDA_CHECK(cudaEventDestroy(stop_event));
    }
};

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace triangles_gpu {
constexpr uint32_t TILE_LOG = 4;
constexpr uint32_t TILE_SIZE = 1 << TILE_LOG;

// Returns bounding box of `triangle`, with exclusive ends.
// TODO: Check whether Cuda array works and whether the 
// function works overall.
__device__ cuda::std::array<int, 4> bounding_box(
    Triangle2D triangle,
    int image_width,
    int image_height
) {
    auto [lowest_x, highest_x] = std::minmax({triangle.a.x, triangle.b.x, triangle.c.x});
    auto [lowest_y, highest_y] = std::minmax({triangle.a.y, triangle.b.y, triangle.c.y});

    // Bound by the image size.
    lowest_x = std::max(lowest_x, 0);
    lowest_y = std::max(lowest_y, 0);
    highest_x = std::min(highest_x, image_width);
    highest_y = std::min(highest_y, image_height);

    return {lowest_x, highest_x, lowest_y, highest_y};
}

// Returns barycentric coordinates for vertex c and b, respectively.
// Uses code adapted from this page: https://blackpawn.com/texts/pointinpoly/
__device__ cuda::std::array<float, 2> barycentric_coordinates(Triangle2D triangle, Point2D point) {
    Point2D v0 = triangle.c - triangle.a;
    Point2D v1 = triangle.b - triangle.a;
    Point2D v2 = point - triangle.a;

    int dot00 = v0.dot(v0);
    int dot01 = v0.dot(v1);
    int dot02 = v0.dot(v2);
    int dot11 = v1.dot(v1);
    int dot12 = v1.dot(v2);

    float invDenom = 1.0 / static_cast<float>(dot00  * dot11 - dot01  * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    float v = (dot00 * dot12 - dot01 * dot02) * invDenom;

    return {u, v};
}

__device__ bool point_inside_triangle(Triangle2D triangle, Point2D point) {
    auto [u, v] = barycentric_coordinates(triangle, point);

    // Determine whether the point is in the triangle.
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

__device__ bool pixel_visible_update(
    Triangle2D triangle,
    Point2D point,
    std::array<float, 3> triangle_z_coords,
    std::vector<float> &max_pixel_z,
    int image_width
) {
    auto [bary_c, bary_b] = barycentric_coordinates(triangle, point);
    std::array<decltype(bary_c), 3> barycentric_coord = {1 - bary_c - bary_b, bary_b, bary_c};

    float pixel_z_value = 0;
    for (int i = 0; i < 3; ++i) {
        pixel_z_value += static_cast<float>(barycentric_coord[i] * triangle_z_coords[i]);
    }

    bool pixel_is_visible = pixel_z_value > max_pixel_z[point.y * image_width + point.x];
    if (pixel_is_visible) {
        max_pixel_z[point.y * image_width + point.x] = pixel_z_value;
    }
    
    return pixel_is_visible;
}

template<uint32_t GROUP_SIZE = 1>
__global__ void block_count_triangles_per_tile(
    int32_t width,
    int32_t height,
    Args args,
    // int32_t n_triangle,
    // float const *triangle_x,      // pointer to GPU memory
    // float const *triangle_y,      // pointer to GPU memory
    // float const *triangle_radius, // pointer to GPU memory
    uint32_t *global_counts
) {
    extern __shared__ uint32_t shmem_tile_counts[];

    // Initialize `shmem_tile_counts` to zero.
    uint32_t num_tiles_x = (width + TILE_SIZE - 1) >> TILE_LOG;
    uint32_t num_tiles_y = (height + TILE_SIZE - 1) >> TILE_LOG;
    uint32_t num_tiles = num_tiles_x * num_tiles_y;
    for (uint32_t i = threadIdx.x; i < num_tiles; i += blockDim.x) {
        shmem_tile_counts[i] = 0;
    }

    __syncthreads();

    // Count how many of the triangles (that this block is responsible for) belong to each tile.
    uint32_t block_start = blockIdx.x * blockDim.x * GROUP_SIZE;
    for (uint32_t i = block_start + threadIdx.x; i < min(block_start + (blockDim.x * GROUP_SIZE), n_triangle); i += blockDim.x) {
        cuda::std::array<Point3D, 3> world_vertices = args.get_triangle(i);
        
        // Project to 2D.
        cuda::std::array<Point2D, 3> screen_triangle_vertices;
        for (int v = 0; v < 3; ++v) {
            float world_x = world_vertices[0];
            float world_y = world_vertices[1];
            
            int screen_x = (world_x + 1.0) * static_cast<double>(width) / 2.0;
            int screen_y = (world_y + 1.0) * static_cast<double>(height) / 2.0;
            screen_triangle_vertices[v] = Point2D({screen_x, screen_y});
        }

        // Create 2D triangle.
        Triangle2D screen_triangle(
            screen_triangle_vertices[0],
            screen_triangle_vertices[1],
            screen_triangle_vertices[2]
        );
        
        // Get all the tiles the triangle covers.
        auto [lowest_x, highest_x, lowest_y, highest_y] = bounding_box(screen_triangle, width, height);
        uint32_t t_tile_start_x = lowest_x >> TILE_LOG;
        uint32_t t_tile_start_y = lowest_y >> TILE_LOG;
        uint32_t t_tile_end_x = highest_x >> TILE_LOG;  // Inclusive
        uint32_t t_tile_end_y = highest_y >> TILE_LOG;  // Inclusive.

        // Increment the tile count for each tile this triangle covers.
        for (uint32_t tile_y = t_tile_start_y; tile_y < t_tile_end_y + 1; ++tile_y) {
            for (uint32_t tile_x = t_tile_start_x; tile_x < t_tile_end_x + 1; ++tile_x) {
                uint32_t one_dim_coord = tile_y * num_tiles_x + tile_x;
                atomicAdd(shmem_tile_counts + one_dim_coord, 1);
            }
        }
    }

    __syncthreads();

    // Write results to global memory.
    for (uint32_t t = threadIdx.x; t < num_tiles; t += blockDim.x) {
        global_counts[t * gridDim.x + blockIdx.x] = shmem_tile_counts[t];
    }
}

__global__ void prefix_sum_and_max_tile_counts(
    int32_t width,
    int32_t height,
    uint32_t *global_counts,
    uint32_t global_counts_size,
    uint32_t group_size,
    uint32_t *max
) {
    uint32_t start = (blockIdx.x * blockDim.x + threadIdx.x) * group_size;

    // Find out why removing this causes a memory access error for only the
    // `atomicMax` call.
    if (start >= global_counts_size) { return; }

    uint32_t accumulator = 0;
    for (int i = start; i < min(start + group_size, global_counts_size); ++i) {
        uint32_t temp = global_counts[i];
        global_counts[i] = accumulator;
        accumulator += temp;
    }

    // Find the max amount of elements in any given tile.
    atomicMax(max, accumulator);
}

__global__ void map_tiles_to_triangles(
    // Image dims.
    uint32_t width,
    uint32_t height,

    // Triangle stuff.
    Args args,

    // Metadata and output.
    uint32_t *per_tile_prefix_sum,
    uint32_t *tiles_to_triangles,
    uint32_t tile_allocation_size,

    // Launch parameters.
    uint32_t group_size
) {
    // Initialize `shmem_tile_counts` to zero.
    uint32_t num_tiles_x = (width + TILE_SIZE - 1) >> TILE_LOG;
    uint32_t num_tiles_y = (height + TILE_SIZE - 1) >> TILE_LOG;
    uint32_t num_tiles = num_tiles_x * num_tiles_y;
    
    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t thread_start = global_index * group_size;
    if (thread_start >= n_triangle) { return; }

    for (uint32_t i = thread_start; i < min(thread_start + group_size, n_triangle); ++i) {
        cuda::std::array<Point3D, 3> world_vertices = args.get_triangle(i);
        
        // Project to 2D.
        cuda::std::array<Point2D, 3> screen_triangle_vertices;
        for (int v = 0; v < 3; ++v) {
            float world_x = world_vertices[0];
            float world_y = world_vertices[1];
            
            int screen_x = (world_x + 1.0) * static_cast<double>(width) / 2.0;
            int screen_y = (world_y + 1.0) * static_cast<double>(height) / 2.0;
            screen_triangle_vertices[v] = Point2D({screen_x, screen_y});
        }

        // Create 2D triangle.
        Triangle2D screen_triangle(
            screen_triangle_vertices[0],
            screen_triangle_vertices[1],
            screen_triangle_vertices[2]
        );
        
        // Get all the tiles the triangle covers.
        auto [lowest_x, highest_x, lowest_y, highest_y] = bounding_box(screen_triangle, width, height);
        uint32_t t_tile_start_x = lowest_x >> TILE_LOG;
        uint32_t t_tile_start_y = lowest_y >> TILE_LOG;
        uint32_t t_tile_end_x = highest_x >> TILE_LOG;  // Inclusive
        uint32_t t_tile_end_y = highest_y >> TILE_LOG;  // Inclusive.

        // Increment the tile count for each tile this triangle covers.
        for (uint32_t tile_y = t_tile_start_y; tile_y < t_tile_end_y + 1; ++tile_y) {
            for (uint32_t tile_x = t_tile_start_x; tile_x < t_tile_end_x + 1; ++tile_x) {
                uint32_t tile_index = tile_y * num_tiles_x + tile_x;
                uint32_t prefix_sum_index = tile_index * ((n_triangle + group_size - 1) / group_size) + global_index;
                uint32_t write_offset = per_tile_prefix_sum[prefix_sum_index];
                per_tile_prefix_sum[prefix_sum_index] = write_offset + 1;
                uint32_t one_dim = tile_allocation_size * tile_index + write_offset;
                tiles_to_triangles[one_dim] = i;
            }
        }
    }
}

__global__ void gpu_print(uint32_t *map, uint32_t num, uint32_t tile_size) {
    if (!(blockIdx.x + threadIdx.x == 0)) { return; }
    for (int row = 0; row < (num / tile_size); ++row) {
        // printf("\n-----------------------------\nTile: %i\n", row);
        for (int col = 0; col < tile_size; ++col) {
            int one_dim = row * tile_size + col;
            printf("%i, ", map[one_dim]);
        }
    }
}

__global__ void draw_triangles(
    int32_t width,
    int32_t height,
    Args args,
    uint32_t *map,
    uint32_t max_triangles_per_tile
) {
    uint32_t x = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint32_t y = blockIdx.y * TILE_SIZE + threadIdx.y;

    uint32_t tile_index = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tile_triangles_start = tile_index * max_triangles_per_tile;

    int32_t pixel_idx = y * width + x;
    for (int i = tile_triangles_start;
         (i < tile_triangles_start + max_triangles_per_tile) && (map[i] < n_triangle);
         ++i
    ) {
        cuda::std::array<Point3D, 3> world_vertices = args.get_triangle(i);

        // Project to 2D and save z-coordinate.
        cuda::std::array<Point3D, 3> world_triangle_vertices;
        cuda::std::array<Point2D, 3> screen_triangle_vertices;
        cuda::std::array<float, 3> z_coords;
        for (int v = 0; v < 3; ++v) {
            float world_x = world_vertices[0];
            float world_y = world_vertices[1];
            float world_z = world_vertices[2];
            world_triangle_vertics[v] = Point3D({world_x, world_y, world_z});
            
            int screen_x = (world_x + 1.0) * static_cast<double>(width) / 2.0;
            int screen_y = (world_y + 1.0) * static_cast<double>(height) / 2.0;
            screen_triangle_vertices[v] = Point2D({screen_x, screen_y});

            z_coords[v] = world_z;
        }
        
        // Create 2D triangle.
        Triangle2D screen_triangle(
            screen_triangle_vertices[0],
            screen_triangle_vertices[1],
            screen_triangle_vertices[2]
        );
        
        Point2D pixel({x, y});
        if (
            point_inside_triangle(triangle, pixel)
            && pixel_visible_update(
                triangle,
                pixel,
                triangle_z_coords,
                max_pixel_z,
                width
            )
        ) {
            Triangle3D world_triangle(
                world_triangle_vertices[0],
                world_triangle_vertices[1],
                world_triangle_vertices[2]
            );

            Point3D norm = world_triangle.normal();
            float light_intensity = norm.dot(light_direction);
            if (light_intensity > 0) {
                float adjusted_value = 255 * light_intensity;
                TGAColor color(
                    adjusted_value,
                    adjusted_value,
                    adjusted_value,
                    255
                );

                args.img.set(x, y, color); 
            } 
        }
    }
}

void launch_render(GpuMemoryPool &memory_pool) {
    // Get buffers
    std::string obj_file = "data/head.obj";
    std::string mtl_path = "data/";
    std::string fbx_file = "data/views.fbx";
    Args args;
    args.get_args(obj_file.c_str(), mtl_path.c_str(), fbx_file.c_str());

    uint32_t num_pixels = FRAME_WIDTH * FRAME_HEIGHT;

    float max_pixel_z = memory_pool.alloc(FRAME_WIDTH * FRAME_HEIGHT * sizeof(float));
    cudaMemset((void*)max_pixel_z, 0xFFFF, num_pixels * sizeof(float));

    constexpr uint32_t GROUP_SIZE = 1;
    constexpr uint32_t C_NUM_THREADS = 32 * 16;
    constexpr uint32_t ELEMENTS_PER_BLOCK = GROUP_SIZE * C_NUM_THREADS;
    uint32_t c_num_blocks = (n_triangle + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    uint32_t num_tiles_x = ((width + TILE_SIZE - 1) >> TILE_LOG);
    uint32_t num_tiles_y = ((height + TILE_SIZE - 1) >> TILE_LOG);
    uint32_t num_tiles = num_tiles_x * num_tiles_y;
    size_t counters_size_in_bytes = sizeof(uint32_t) * num_tiles * c_num_blocks;
    size_t counters_size = counters_size_in_bytes / sizeof(uint32_t);
    uint32_t *counters_d = reinterpret_cast<uint32_t *>(memory_pool.alloc(counters_size_in_bytes));

    Timer t;

    // MARK: Count how many triangles in each block belong to each tile.

    t.start();
    uint32_t shmem_size = num_tiles * sizeof(uint32_t);
    block_count_triangles_per_tile<<<c_num_blocks, C_NUM_THREADS, shmem_size>>>(
        width,
        height,
        args,
        counters_d
    );
    t.stop("Counter time:");

    // MARK: Prefix scan to find out how many triangles in total belong to each tile.

    t.start();

    // Get memory locations and initialize max values.
    uint32_t most_triangles_in_tile_h = 0;
    uint32_t *most_triangles_in_tile_d =
        reinterpret_cast<uint32_t *>(memory_pool.alloc(sizeof(most_triangles_in_tile_h)));
    CUDA_CHECK(cudaMemcpy(
        most_triangles_in_tile_d,
        &most_triangles_in_tile_h,
        sizeof(uint32_t), // number of bytes to copy
        cudaMemcpyHostToDevice
    ));

    // TODO: Figure out why the optimal number for `scan_threads` is so low.
    uint32_t scan_threads = 32 * 1;
    uint32_t scan_per_block = scan_threads * c_num_blocks;
    uint32_t scan_blocks = (counters_size + scan_per_block - 1) / scan_per_block;
    prefix_sum_and_max_tile_counts<<<scan_blocks, scan_threads>>>(
        width,
        height,
        counters_d,
        counters_size,
        c_num_blocks,
        most_triangles_in_tile_d);

    CUDA_CHECK(cudaMemcpy(
        &most_triangles_in_tile_h, // pointer to CPU memory
        most_triangles_in_tile_d, // pointer to GPU memory
        sizeof(uint32_t), // number of bytes to copy
        cudaMemcpyDeviceToHost
    ));

    t.stop("Scan time:");

    t.start();

    uint32_t map_size_bytes = most_triangles_in_tile_h * num_tiles * sizeof(uint32_t);
    uint32_t *map = reinterpret_cast<uint32_t *>(memory_pool.alloc(map_size_bytes));
    // Timer t;
    // t.start();
    cudaMemset(static_cast<void *>(map), 0xFF, map_size_bytes);
    // t.stop("Test");
    uint32_t map_threads = 32 * 4;
    uint32_t map_elements_per_block = map_threads * ELEMENTS_PER_BLOCK;
    uint32_t map_blocks = (n_triangle + map_elements_per_block - 1) / map_elements_per_block;
    map_tiles_to_triangles<<<map_blocks, map_threads>>>(
        width,
        height,

        args,

        counters_d,
        map,
        most_triangles_in_tile_h,

        ELEMENTS_PER_BLOCK
    );

    // gpu_print<<<1,1>>>(map, map_size_bytes >> 2, most_triangles_in_tile_h);
    t.stop("Tile to blocks time:");

    t.start();
    draw_triangles<<<dim3(num_tiles_x, num_tiles_y), dim3(TILE_SIZE, TILE_SIZE)>>>(
        width,
        height,
        args,
        map,
        most_triangles_in_tile_h
    );

    // std::cout << num_tiles 
    t.stop("Draw");

    // printf("\n-----------------------\n");

    // DO NOT DELETE.
    args.clean_args();
}

} // namespace triangles_gpu
