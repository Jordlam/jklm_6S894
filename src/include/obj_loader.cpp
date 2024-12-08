#include "obj_loader.h"
#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "stb_image.h"

void Model::loadModel(const std::string &obj_file, const std::string &mtl_path) {
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_file.c_str(), mtl_path.c_str(), 0, 0);

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
    printf("We parsed %li vertices...\n", attrib.vertices.size());
    printf("We parsed %li normals...\n", attrib.normals.size());
    printf("We parsed %li texcoords...\n", attrib.texcoords.size());
    printf("We parsed %li faces...\n", nfaces(0));
    printf("We parsed %li materials...\n", materials.size());
    printf("Example...\n");
    std::vector<float> vertices = vert(0, 0, 0);
    float world_x = vertices[0];
    float world_y = vertices[1];
    float world_z = vertices[2];
    printf("Coordinates %f, %f, %f\n", world_x, world_y, world_z);
}
size_t Model::nfaces(int i) {
    return shapes[i].mesh.indices.size();
}
std::vector<float> Model::vert(int i, int t, int v) {
    tinyobj::index_t idx = shapes[i].mesh.indices[t + v];
    int vertex_index = idx.vertex_index;
    float v_x = attrib.vertices[3 * vertex_index + 0];
    float v_y = attrib.vertices[3 * vertex_index + 1];
    float v_z = attrib.vertices[3 * vertex_index + 2];
    return {v_x, v_y, v_z};
}

// bool WriteObj(const std::string& filename, const tinyobj::attrib_t& attributes, const std::vector<tinyobj::shape_t>& shapes, const std::vector<tinyobj::material_t>& materials, bool coordTransform) {
//   FILE* fp = fopen(filename.c_str(), "w");
//   if (!fp) {
//     fprintf(stderr, "Failed to open file [ %s ] for write.\n", filename.c_str());
//     return false;
//   }

//   std::string basename = GetFileBasename(filename);
//   std::string material_filename = basename + ".mtl";

//   int prev_material_id = -1;

//   fprintf(fp, "mtllib %s\n\n", material_filename.c_str());

//   // facevarying vtx
//   for (size_t k = 0; k < attributes.vertices.size(); k+=3) {
//     if (coordTransform) {
//       fprintf(fp, "v %f %f %f\n",
//         attributes.vertices[k + 0],
//         attributes.vertices[k + 2],
//         -attributes.vertices[k + 1]);
//     } else {
//       fprintf(fp, "v %f %f %f\n",
//         attributes.vertices[k + 0],
//         attributes.vertices[k + 1],
//         attributes.vertices[k + 2]);
//     }
//   }

//   fprintf(fp, "\n");

//   // facevarying normal
//   for (size_t k = 0; k < attributes.normals.size(); k += 3) {
//     if (coordTransform) {
//       fprintf(fp, "vn %f %f %f\n",
//         attributes.normals[k + 0],
//         attributes.normals[k + 2],
//         -attributes.normals[k + 1]);
//     } else {
//       fprintf(fp, "vn %f %f %f\n",
//         attributes.normals[k + 0],
//         attributes.normals[k + 1],
//         attributes.normals[k + 2]);
//     }
//   }

//   fprintf(fp, "\n");

//   // facevarying texcoord
//   for (size_t k = 0; k < attributes.texcoords.size(); k += 2) {
//     fprintf(fp, "vt %f %f\n",
//       attributes.texcoords[k + 0],
//       attributes.texcoords[k + 1]);
//   }

//   for (size_t i = 0; i < shapes.size(); i++) {
//     fprintf(fp, "\n");

//     if (shapes[i].name.empty()) {
//       fprintf(fp, "g Unknown\n");
//     } else {
//       fprintf(fp, "g %s\n", shapes[i].name.c_str());
//     }

//     bool has_vn = false;
//     bool has_vt = false;
//     // Assumes normals and textures are set shape-wise.
//     if(shapes[i].mesh.indices.size() > 0){
//       has_vn = shapes[i].mesh.indices[0].normal_index != -1;
//       has_vt = shapes[i].mesh.indices[0].texcoord_index != -1;
//     }

//     // face
//     int face_index = 0;
//     for (size_t k = 0; k < shapes[i].mesh.indices.size(); k += shapes[i].mesh.num_face_vertices[face_index++]) {
//       // Check Materials
//       int material_id = shapes[i].mesh.material_ids[face_index];
//       if (material_id != prev_material_id) {
//         std::string material_name = materials[material_id].name;
//         fprintf(fp, "usemtl %s\n", material_name.c_str());
//         prev_material_id = material_id;
//       }

//       unsigned char v_per_f = shapes[i].mesh.num_face_vertices[face_index];
//       // Imperformant, but if you want to have variable vertices per face, you need some kind of a dynamic loop.
//       fprintf(fp, "f");
//       for(int l = 0; l < v_per_f; l++){
//         const tinyobj::index_t& ref = shapes[i].mesh.indices[k + l];
//         if(has_vn && has_vt){
//           // v0/t0/vn0
//           fprintf(fp, " %d/%d/%d", ref.vertex_index + 1, ref.texcoord_index + 1, ref.normal_index + 1);
//           continue;
//         }
//         if(has_vn && !has_vt){
//           // v0//vn0
//           fprintf(fp, " %d//%d", ref.vertex_index + 1, ref.normal_index + 1);
//           continue;
//         }
//         if(!has_vn && has_vt){
//           // v0/vt0
//           fprintf(fp, " %d/%d", ref.vertex_index + 1, ref.texcoord_index + 1);
//           continue;
//         }
//         if(!has_vn && !has_vt){
//           // v0 v1 v2
//           fprintf(fp, " %d", ref.vertex_index + 1);
//           continue;
//         }
//       }
//       fprintf(fp, "\n");
//     }
//   }

//   fclose(fp);

//   //
//   // Write material file
//   //
//   bool ret = WriteMat(material_filename, materials);

//   return ret;
// }
