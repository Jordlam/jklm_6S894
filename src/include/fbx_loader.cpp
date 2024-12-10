#include "fbx_loader.h"

void Scene::loadFBX(const std::string &fbx_file) {
    ufbx_load_opts opts = {0};
    ufbx_error error;
    scene = ufbx_load_file(fbx_file.c_str(), &opts, &error);
    if (!scene) {
        printf("Error: Failed to load scene.\n");
        return;
    }

    std::vector<ufbx_node> parsed_cameras;
    std::vector<ufbx_node> parsed_lights;
    for (int i = 0; i < scene->nodes.count; i++) {
        // Parse cameras
        if (scene->nodes.data[i]->camera) {
            
            parsed_cameras.push_back(*(scene->nodes.data[i]));
        }
        // Parse lights
        else if (scene->nodes.data[i]->light) {
            parsed_lights.push_back(*(scene->nodes.data[i]));
        }
    }
    cameras = parsed_cameras;
    lights = parsed_lights;

    // TODO Some tests
    ufbx_vec2 cam_size = get_camera(0).camera->resolution;
    ufbx_vec3 cam_pos = get_camera(0).local_transform.translation;
    ufbx_vec3 light_dir = get_light(0).light->local_direction;
    printf("Example...\n");
    printf("We have %li nodes in our scene.\n", scene->nodes.count);
    printf("Camera view with width %f and height %f\n", cam_size.x, cam_size.y);
    printf("Camera position at %f, %f, %f\n", cam_pos.x, cam_pos.y, cam_pos.z);
    printf("Light vector %f, %f, %f\n", light_dir.x, light_dir.y, light_dir.z);
}
ufbx_node Scene::get_camera(int i) {
    return cameras[i];
}
ufbx_node Scene::get_light(int i) {
    return lights[i];
}
std::vector<int> Scene::project(float x, float y, float z) {
    ufbx_node camera = get_camera(0);
    ufbx_vec3 translation = camera.geometry_transform.translation;
    ufbx_matrix mm = camera.geometry_to_world;
    float new_x = x*mm.m00 + y*mm.m10 + z*mm.m20 + translation.x;
    float new_y = x*mm.m01 + y*mm.m11 + z*mm.m21 + translation.y;
    float new_z = x*mm.m02 + y*mm.m12 + z*mm.m22 + translation.z;
    float new_w = x*0 + y*0 - z*1 + 1*0;
    if (new_w == 0) {
        printf("Error: We encountered w'=0\n");
    }
    new_x /= new_w;
    new_y /= new_w;
    new_z /= new_w;
    return {(int)new_x, (int)new_y, (int)new_z};

}
void Scene::cleanup() {
    ufbx_free_scene(scene);
}

void example(const std::string &fbx_file) {
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
