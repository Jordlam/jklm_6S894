#ifndef FBX_LOADER
#define FBX_LOADER

#include "ufbx.h"
#include <string>
#include <vector>

class Scene {
private:
    ufbx_scene* scene;
    std::vector<ufbx_node> cameras;
    std::vector<ufbx_node> lights;
public:
    void loadFBX(const std::string &fbx_file);
    ufbx_node get_camera(int i);
    ufbx_node get_light(int i);
    // As described here: https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix.html
    std::vector<int> project(float x, float y, float z);
    void cleanup();
};

// Example. Don't use
void example(const std::string &fbx_file);

#endif
