#define TINYOBJLOADER_IMPLEMENTATION
#include "include/tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#include <vector>

void loadModel(const std::string &obj_file, const std::string &mtl_file) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;

    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_file.c_str(), mtl_file.c_str(), 0, 0);

    if (!warn.empty()) {
        printf("Warning: %s\n", warn.c_str());
    }
    if (!err.empty()) {
        printf("Error: %s\n", err.c_str());
    }
    if (!ret) {
        printf("Error loading OBJ file:\n");
        return;
    }

    // Parse textures
    if (materials.size() != 1) {
        printf("Error: object does not have a single material.\n");
        return;
    }
    tinyobj::real_t Ka[3];
    Ka[0] = materials[0].ambient[0];
    Ka[1] = materials[0].ambient[1];
    Ka[2] = materials[0].ambient[2];
    tinyobj::real_t Kd[3];
    Kd[0] = materials[0].diffuse[0];
    Kd[1] = materials[0].diffuse[1];
    Kd[2] = materials[0].diffuse[2];
    tinyobj::real_t Ks[3];
    Ks[0] = materials[0].specular[0];
    Ks[1] = materials[0].specular[1];
    Ks[2] = materials[0].specular[2];
    tinyobj::real_t Ke[3];
    Ke[0] = materials[0].emission[0];
    Ke[1] = materials[0].emission[1];
    Ke[2] = materials[0].emission[2];
    tinyobj::real_t Ns = materials[0].shininess;
    tinyobj::real_t Ni = materials[0].ior;
    tinyobj::real_t d = materials[0].dissolve;
    int illum = materials[0].illum;
    std::string map_Kd = materials[0].diffuse_texname;
    // TODO now do something with these fields
    printf("Example textures:\n");
    printf("%f, %f, %f\n", Ka[0], Ka[1], Ka[2]);
    printf("%s\n", map_Kd.c_str());

    // Load texture map
    int width, height, channels;
    unsigned char *data = stbi_load(map_Kd.c_str(), &width, &height, &channels, 0);
    // TODO now do something with the texture map
    // You can use the below texcoords to index into this data
    printf("Example texture map:\n");
    printf("%i, %i, %i\n", width, height, channels);

    // Parse vertices, normals, and texture coordinates
    for (size_t f = 0; f < shapes.size(); f++) {
        size_t index_offset = 0;
        const tinyobj::shape_t shape = shapes[f];
        for (size_t i = 0; i < shape.mesh.num_face_vertices.size(); i++) {
            int num_vertices = shape.mesh.num_face_vertices[i];
            if (num_vertices != 3) {
                printf("Error: polygons were not triangulated.\n");
                return;
            }
            for (int v = 0; v < num_vertices; v++) {
                tinyobj::index_t idx = shape.mesh.indices[v];
                int vertex_index = idx.vertex_index;
                int normal_index = idx.normal_index;
                int texcoord_index = idx.texcoord_index;

                // Get vertices
                float v_x = attrib.vertices[3 * vertex_index + 0];
                float v_y = attrib.vertices[3 * vertex_index + 1];
                float v_z = attrib.vertices[3 * vertex_index + 2];
                // Get textures
                float vt_x = attrib.texcoords[2 * texcoord_index + 0];
                float vt_y = attrib.texcoords[2 * texcoord_index + 1];
                // Get normals
                float vn_x = attrib.normals[3 * normal_index + 0];
                float vn_y = attrib.normals[3 * normal_index + 1];
                float vn_z = attrib.normals[3 * normal_index + 2];
                // TODO now do something with these fields
                // For simple models they should all be the same smoothing group
                // int s = 1;
                printf("Example vertices:\n");
                printf("%f, %f, %f\n", v_x, v_y, v_z);
                printf("%f, %f\n", vt_x, vt_y);
                printf("%f, %f, %f\n", vn_x, vn_y, vn_z);
                return;
            }
            index_offset += num_vertices;
        }
    }
}

int main() {
    std::string obj_file = "data/amongus_triangles.obj";
    std::string mtl_path = "data";
    loadModel(obj_file, mtl_path);
    return 0;
}