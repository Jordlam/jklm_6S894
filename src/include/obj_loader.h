#ifndef OBJ_LOADER
#define OBJ_LOADER

#include "tiny_obj_loader.h"
#include "stb_image.h"
#include <vector>

class Model {
private:
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
public:
    void loadModel(const std::string &obj_file, const std::string &mtl_path=std::string(""));
    // Return the number of faces for object i
    size_t nfaces(int i);
    tinyobj::attrib_t get_attrib();
    tinyobj::shape_t get_shape(int i);
    std::vector<float> vert(int i, int t, int v);
    void cleanup();
};

// Don't use but for example:
// bool WriteObj(const std::string& filename, const tinyobj::attrib_t& attributes, const std::vector<tinyobj::shape_t>& shapes, const std::vector<tinyobj::material_t>& materials, bool coordTransform);
void example(const std::string &obj_file, const std::string &mtl_file);

#endif
