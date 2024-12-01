#include "include/ufbx.h"
#include <string>

void loadFBX(const std::string &fbx_file) {
    ufbx_load_opts opts = {0};
    ufbx_error error;
    ufbx_scene* scene = ufbx_load_file(fbx_file.c_str(), &opts, &error);

    if (!scene) {
        printf("Error: Failed to load scene.\n");
        return;
    }

    for (int i = 0; i < scene->cameras.count; i++) {
        const ufbx_camera* camera = scene->cameras.data[i];
        // TODO now do something with the camera
        ufbx_real ar = camera->aspect_ratio;
        ufbx_real fl = camera->focal_length_mm;
        ufbx_vec2 res = camera->resolution;
        ufbx_real x = res.x;
        ufbx_real y = res.y;
        printf("%f, %f, %f, %f\n", ar, fl, x, y);
    }

    if (scene->lights.count != 1) {
        printf("Error: Could not parse a single light.\n");
        return;
    }
    const ufbx_light* light = scene->lights.data[0];
    // TODO now do something with the light
    ufbx_real intensity = light->intensity;
    ufbx_light_type lt = light->type;
    std::string lt_str;
    switch (lt) {
        case ufbx_light_type::UFBX_LIGHT_POINT:
            lt_str = "UFBX_LIGHT_POINT";
            break;
        case ufbx_light_type::UFBX_LIGHT_DIRECTIONAL:
            lt_str = "UFBX_LIGHT_DIRECTIONAL";
            break;
        case ufbx_light_type::UFBX_LIGHT_SPOT:
            lt_str = "UFBX_LIGHT_SPOT";
            break;
        case ufbx_light_type::UFBX_LIGHT_AREA:
            lt_str = "UFBX_LIGHT_AREA";
            break;
        case ufbx_light_type::UFBX_LIGHT_VOLUME:
            lt_str = "UFBX_LIGHT_VOLUME";
            break;
        default:
            lt_str = "We could not find the specific light source.";
    }
    printf("%f, %s\n", intensity, lt_str.c_str());

    // Clean up
    ufbx_free_scene(scene);
}

int main() {
    std::string fbx_file = "data/views.fbx";
    loadFBX(fbx_file);
    return 0;
}