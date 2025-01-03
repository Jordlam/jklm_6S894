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

// New Additions.
#include <cuda/std/array>


// #include "cpu_rasterizer.h"
#include "args.h"


struct Point2D {
    int x;
    int y;

    __device__ Point2D operator+(Point2D p) {
        return {x + p.x, y + p.y};
    }
    __device__ Point2D operator-(Point2D p) {
        return {x - p.x, y - p.y};
    }

    __device__ int dot(Point2D p) {
        return x * p.x + y * p.y;
    }
};

struct Point3D {
    float x;
    float y;
    float z;
    // Point3D() : x(0.0f), y(0.0f), z(0.0f) {}  // QUESTION: Why does this cause errors?

    __device__ Point3D operator+(Point3D p) {
        return {x + p.x, y + p.y, z + p.z};
    }

    __device__ Point3D operator-(Point3D p) {
        return {x - p.x, y - p.y, z - p.z};
    }

    __device__ Point3D cross(Point3D p) {
        return {y * p.z - z * p.y, z * p.x - x * p.z, x * p.y  - y  * p.x};
    }

    __device__ float dot(Point3D p) {
        return x * p.x + y * p.y + z * p.z;
    }

    __device__ float magnitude() {
        return sqrt(x * x + y * y + z * z);
    }

    __device__ Point3D normalize() {
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

    __device__ Triangle<Point>(
        Point _a,
        Point _b,
        Point _c
    ) :
        a(_a), b(_b), c(_c)
    {}

    __device__ Point normal();
};

using Triangle2D = Triangle<Point2D>;
using Triangle3D = Triangle<Point3D>;

// `normal` only makes sense for `Triangle3D`.
template<>
__device__ Point3D Triangle3D::normal() {
    // Must be in this order (i.e., using `b` as our point of reference),
    // presumably because the cross product is not commutative, and the
    // other orders cause issues.
    Point3D u = a - b;
    Point3D v = c - b;
    
    // Point3D u = a - c;
    // Point3D v = b - c;

    return u.cross(v).normalize();
}















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

// namespace triangles_gpu {
constexpr uint32_t TILE_LOG = 4;
constexpr uint32_t TILE_SIZE = 1 << TILE_LOG;

__device__ cuda::std::array<Point3D, 3> get_triangle(
    int i,
    int32_t* vertices,
    float *vertex_x,
    float *vertex_y,
    float *vertex_z
) {
    int32_t v0_idx = vertices[3*i + 0];
    int32_t v1_idx = vertices[3*i + 1];
    int32_t v2_idx = vertices[3*i + 2];

    Point3D point0({vertex_x[v0_idx], vertex_y[v0_idx], vertex_z[v0_idx]});
    Point3D point1({vertex_x[v1_idx], vertex_y[v1_idx], vertex_z[v1_idx]});
    Point3D point2({vertex_x[v2_idx], vertex_y[v2_idx], vertex_z[v2_idx]});

    return {point0, point1, point2};
}


// Returns bounding box of `triangle`, with exclusive ends.
// TODO: Check whether Cuda array works and whether the
// function works overall.
__device__ cuda::std::array<int, 4> bounding_box(
    Triangle2D triangle,
    int image_width,
    int image_height
) {
    // auto [lowest_x, highest_x] = std::minmax({triangle.a.x, triangle.b.x, triangle.c.x});
    // auto [lowest_y, highest_y] = std::minmax({triangle.a.y, triangle.b.y, triangle.c.y});

    int lowest_x = min(min(triangle.a.x, triangle.b.x), triangle.c.x);
    int highest_x = max(max(triangle.a.x, triangle.b.x), triangle.c.x);
    int lowest_y = min(min(triangle.a.y, triangle.b.y), triangle.c.y);
    int highest_y = max(max(triangle.a.y, triangle.b.y), triangle.c.y);

    // // Bound by the image size.
    lowest_x = max(lowest_x, 0);
    lowest_y = max(lowest_y, 0);
    highest_x = min(highest_x, image_width);
    highest_y = min(highest_y, image_height);

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

template<uint32_t WIDTH, uint32_t HEIGHT>
__device__ bool pixel_visible_update(
    Triangle2D triangle,
    Point2D point,
    cuda::std::array<float, 3> triangle_z_coords,
    float *max_pixel_z
) {
    auto [bary_c, bary_b] = barycentric_coordinates(triangle, point);
    cuda::std::array<decltype(bary_c), 3> barycentric_coord = {1 - bary_c - bary_b, bary_b, bary_c};

    float pixel_z_value = 0;
    for (int i = 0; i < 3; ++i) {
        pixel_z_value += static_cast<float>(barycentric_coord[i] * triangle_z_coords[i]);
    }

    bool pixel_is_visible = pixel_z_value > max_pixel_z[point.y * WIDTH + point.x];
    if (pixel_is_visible) {
        max_pixel_z[point.y * WIDTH + point.x] = pixel_z_value;
    }

    return pixel_is_visible;
}

template<uint32_t GROUP_SIZE = 1>
__global__ void block_count_triangles_per_tile(
    int32_t width,
    int32_t height,
    
    int32_t n_triangle,
    int32_t *vertices,
    float *vertex_x,
    float *vertex_y,
    float *vertex_z,

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
        cuda::std::array<Point3D, 3> world_vertices = get_triangle(i, vertices, vertex_x, vertex_y, vertex_z);

        // Project to 2D.
        cuda::std::array<Point2D, 3> screen_triangle_vertices;
        for (int v = 0; v < 3; ++v) {
            float world_x = world_vertices[v].x;
            float world_y = world_vertices[v].y;

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
    int32_t n_triangle,
    int32_t *vertices,
    float *vertex_x,
    float *vertex_y,
    float *vertex_z,


    // Metadata and output.
    uint32_t *per_tile_prefix_sum,
    uint32_t *tiles_to_triangles,
    uint32_t tile_allocation_size,

    // Launch parameters.
    uint32_t group_size
) {
    // Initialize `shmem_tile_counts` to zero.
    uint32_t num_tiles_x = (width + TILE_SIZE - 1) >> TILE_LOG;
    // uint32_t num_tiles_y = (height + TILE_SIZE - 1) >> TILE_LOG;
    // uint32_t num_tiles = num_tiles_x * num_tiles_y;

    const uint32_t global_index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t thread_start = global_index * group_size;
    if (thread_start >= n_triangle) { return; }

    for (uint32_t i = thread_start; i < min(thread_start + group_size, n_triangle); ++i) {
        cuda::std::array<Point3D, 3> world_vertices = get_triangle(i, vertices, vertex_x, vertex_y, vertex_z);

        // Project to 2D.
        cuda::std::array<Point2D, 3> screen_triangle_vertices;
        for (int v = 0; v < 3; ++v) {
            float world_x = world_vertices[v].x;
            float world_y = world_vertices[v].y;

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

template<uint32_t WIDTH, uint32_t HEIGHT>
__global__ void draw_triangles(
    int32_t n_triangle,
    int32_t *vertices,
    float *vertex_x,
    float *vertex_y,
    float *vertex_z,
    float *z_buffer,
    float *pixel_intensity,

    uint32_t *map,
    uint32_t max_triangles_per_tile
) {
    uint32_t x = blockIdx.x * TILE_SIZE + threadIdx.x;
    uint32_t y = blockIdx.y * TILE_SIZE + threadIdx.y;

    uint32_t tile_index = blockIdx.y * gridDim.x + blockIdx.x;
    uint32_t tile_triangles_start = tile_index * max_triangles_per_tile;

    int32_t pixel_idx = y * WIDTH + x;

    if (pixel_idx >= 1024 * 1024) { return; }


    for (int i = tile_triangles_start;
         (i < tile_triangles_start + max_triangles_per_tile) && (map[i] < n_triangle);
         ++i
    ) {
        cuda::std::array<Point3D, 3> world_vertices = get_triangle(i, vertices, vertex_x, vertex_y, vertex_z);

        // Project to 2D and save z-coordinate.
        cuda::std::array<Point3D, 3> world_triangle_vertices;
        cuda::std::array<Point2D, 3> screen_triangle_vertices;
        cuda::std::array<float, 3> z_coords;
        for (int v = 0; v < 3; ++v) {
            float world_x = world_vertices[v].x;
            float world_y = world_vertices[v].y;
            float world_z = world_vertices[v].z;
            world_triangle_vertices[v] = Point3D({world_x, world_y, world_z});

            int screen_x = (world_x + 1.0) * static_cast<double>(WIDTH) / 2.0;
            int screen_y = (world_y + 1.0) * static_cast<double>(HEIGHT) / 2.0;
            screen_triangle_vertices[v] = Point2D({screen_x, screen_y});

            z_coords[v] = world_z;
        }

        // Create 2D triangle.
        Triangle2D screen_triangle(
            screen_triangle_vertices[0],
            screen_triangle_vertices[1],
            screen_triangle_vertices[2]
        );

        Point2D pixel({static_cast<int>(x), static_cast<int>(y)});

        bool o = point_inside_triangle(screen_triangle, pixel);
        bool b = pixel_visible_update<WIDTH, HEIGHT>(
            screen_triangle,
            pixel,
            z_coords,
            z_buffer
        );

        if (o && b
            // point_inside_triangle(screen_triangle, pixel)
            // && pixel_visible_update<WIDTH, HEIGHT>(
            //     screen_triangle,
            //     pixel,
            //     z_coords,
            //     z_buffer
            // )
        ) {

            Triangle3D world_triangle(
                world_triangle_vertices[0],
                world_triangle_vertices[1],
                world_triangle_vertices[2]
            );

            Point3D norm = world_triangle.normal();

            // TODO: Replace hardcoded light direction.
            Point3D light_direction({0.0, 0.0, -1.0});
            float light_intensity = norm.dot(light_direction);
            if (light_intensity > 0) {
                float adjusted_value = 255.0f * light_intensity;
                // printf("Here123: %i", pixel_idx);
                pixel_intensity[pixel_idx] = adjusted_value;
            }
        }
    }
}

void launch_render(GpuMemoryPool &memory_pool) {
    constexpr uint32_t width = 1024;
    constexpr uint32_t height = 1024;
    constexpr uint32_t num_pixels = width * height;

    // Get buffers
    std::string obj_file = "data/head.obj";
    std::string mtl_path = "data/";
    std::string fbx_file = "data/views.fbx";
    Args args;
    args.get_args(obj_file.c_str(), mtl_path.c_str(), fbx_file.c_str());

    int32_t n_triangle = args.n_triangle;

    // Allocate device memory.
    uint32_t vertices_size = sizeof(int32_t) * n_triangle * 3;
    uint32_t vertex_dim = sizeof(float) * n_triangle * 3;
    // int32_t* vertices_d = (int32_t*)memory_pool.alloc(vertices_size);
    // float* vertex_x_d = (float*)memory_pool.alloc(vertex_dim);
    // float* vertex_y_d = (float*)memory_pool.alloc(vertex_dim);
    // float* vertex_z_d = (float*)memory_pool.alloc(vertex_dim);
    // float* light_intensity_d = (float*)memory_pool.alloc(num_pixels * sizeof(float));

    // float* z_buffer_d = (float*)memory_pool.alloc(sizeof(float) * num_pixels);

    int32_t* vertices_d = nullptr;
    int x = cudaMalloc(&vertices_d, vertices_size);
    printf("1: %i\n", x);
    
    float* vertex_x_d = nullptr;
    x = cudaMalloc(&vertex_x_d, vertex_dim);
    printf("1: %i\n", x);
    
    float* vertex_y_d = nullptr;
    x = cudaMalloc(&vertex_y_d, vertex_dim);
    printf("1: %i\n", x);
    
    float* vertex_z_d = nullptr;
    x = cudaMalloc(&vertex_z_d, vertex_dim);
    printf("1: %i\n", x);

    float* light_intensity_d = nullptr;
    x = cudaMalloc(&light_intensity_d, num_pixels * sizeof(float));
    printf("1: %i\n", x);

    float* z_buffer_d = nullptr;
    x = cudaMalloc(&z_buffer_d, sizeof(float) * num_pixels);
    printf("1: %i\n", x);


    float *z_buffer_h = new float[num_pixels];
    for (int i = 0; i < num_pixels; ++i) {
        z_buffer_h[i] = std::numeric_limits<float>::lowest();
    }

    cudaMemcpy(
        z_buffer_d,
        z_buffer_h,
        num_pixels * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // Copy to device memory.
    CUDA_CHECK(cudaMemcpy(
        vertices_d,
        args.vertices,
        vertices_size, // number of bytes to copy
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        vertex_x_d,
        args.vertex_x,
        vertex_dim, // number of bytes to copy
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        vertex_y_d,
        args.vertex_y,
        vertex_dim, // number of bytes to copy
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        vertex_z_d,
        args.vertex_z,
        vertex_dim, // number of bytes to copy
        cudaMemcpyHostToDevice
    )); 

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

    // Timer t;

    // MARK: Count how many triangles in each block belong to each tile.

    // t.start();
    uint32_t shmem_size = num_tiles * sizeof(uint32_t);

    // printf("c_num_blocks: %i, c_num_threads: %i, shmem_size: %i\n", c_num_blocks, C_NUM_THREADS, shmem_size);
    block_count_triangles_per_tile<<<c_num_blocks, C_NUM_THREADS, shmem_size>>>(
        width,
        height,

        args.n_triangle,
        vertices_d,
        vertex_x_d,
        vertex_y_d,
        vertex_z_d,
        
        counters_d
    );

    // printf("After123\n");

    uint32_t *counters_h = (uint32_t *)malloc(counters_size_in_bytes);

    CUDA_CHECK(cudaMemcpy(
        counters_h, // pointer to CPU memory
        counters_d, // pointer to GPU memory
        counters_size_in_bytes, // number of bytes to copy
        cudaMemcpyDeviceToHost
    ));

    // t.stop("Counter time:");

    // MARK: Prefix scan to find out how many triangles in total belong to each tile.

    // t.start();

    // Get memory locations and initialize max values.
    uint32_t most_triangles_in_tile_h = 0;
    // uint32_t *most_triangles_in_tile_d =
    // reinterpret_cast<uint32_t *>(memory_pool.alloc(sizeof(most_triangles_in_tile_h)));
    uint32_t *most_triangles_in_tile_d = nullptr;
    x = cudaMalloc(&most_triangles_in_tile_d, sizeof(most_triangles_in_tile_h));
    

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

    // t.stop("Scan time:");

    // t.start();

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

        args.n_triangle,
        vertices_d,
        vertex_x_d,
        vertex_y_d,
        vertex_z_d,

        counters_d,
        map,
        most_triangles_in_tile_h,

        ELEMENTS_PER_BLOCK
    );

    // gpu_print<<<1,1>>>(map, map_size_bytes >> 2, most_triangles_in_tile_h);
    // t.stop("Tile to blocks time:");

    // t.start();
    draw_triangles<width, height><<<dim3(num_tiles_x, num_tiles_y), dim3(TILE_SIZE, TILE_SIZE)>>>(
        args.n_triangle,
        vertices_d,
        vertex_x_d,
        vertex_y_d,
        vertex_z_d,
        z_buffer_d,
        light_intensity_d,

        map,
        most_triangles_in_tile_h
    );

    // return;
    
    float* light_values_h = new float[num_pixels];
    cudaMemcpy(
        light_values_h,
        light_intensity_d,
        num_pixels * sizeof(float), // number of bytes to copy
        cudaMemcpyDeviceToHost
    );


    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint32_t pixel_one_dim = y * width + x;
            float light_value = light_values_h[pixel_one_dim];
            if (pixel_one_dim < 100) {
                printf("Light value: %f\n", light_value);
            }
            TGAColor color(
                light_value,
                light_value,
                light_value,
                255
            );

            args.img.set(x, y, color);
        }
    }


    args.img.write_tga_file("out/framebuffer.tga");

    // std::cout << num_tiles
    // t.stop("Draw");

    // printf("\n-----------------------\n");

    // DO NOT DELETE.
    args.clean_args();
    delete light_values_h;
    delete z_buffer_h;

}

// } // namespace triangles_gpu


GpuMemoryPool::~GpuMemoryPool() {
    // for (auto ptr : allocations_) {
    //     CUDA_CHECK(cudaFree(ptr));
    // }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
    }
}

int main() {
    auto memory_pool = GpuMemoryPool();
    launch_render(memory_pool);
    return 0;
}