// TAKEN FROM THE TINYRENDERER SOURCE CODE.

#pragma once
#include <cstdint>
#include <fstream>
#include <vector>

#pragma pack(push,1)
struct TGAHeader {
    std::uint8_t  idlength = 0;
    std::uint8_t  colormaptype = 0;
    std::uint8_t  datatypecode = 0;
    std::uint16_t colormaporigin = 0;
    std::uint16_t colormaplength = 0;
    std::uint8_t  colormapdepth = 0;
    std::uint16_t x_origin = 0;
    std::uint16_t y_origin = 0;
    std::uint16_t width = 0;
    std::uint16_t height = 0;
    std::uint8_t  bitsperpixel = 0;
    std::uint8_t  imagedescriptor = 0;
};
#pragma pack(pop)

struct TGAColor {
    std::uint8_t bgra[4] = {0,0,0,0};


    // Added by Keawe Mann.
    //
    // These constructors were oddly removed in one of the TinyRenderer
    // commits, but I just added them back. 
    TGAColor() = default;
    TGAColor(const std::uint8_t R, const std::uint8_t G, const std::uint8_t B, const std::uint8_t A=255) : bgra{B,G,R,A}, bytespp(4) { }
    TGAColor(const std::uint8_t *p, const std::uint8_t bpp) : bytespp(bpp) {
        for (int i=bpp; i--; bgra[i] = p[i]);
    }

    // Added by Keawe Mann.
    // Not sure if this existed anywhere before, but
    // the code assumed it existed.
    TGAColor(
        const std::uint8_t R,
        const std::uint8_t G,
        const std::uint8_t B,
        const std::uint8_t A,
        const std::uint8_t bpp
    ) : 
        bgra{B,G,R,A},
        bytespp(bpp)
    { }
    
    std::uint8_t bytespp = 4;
    std::uint8_t& operator[](const int i) { return bgra[i]; }
};

struct TGAImage {
    enum Format { GRAYSCALE=1, RGB=3, RGBA=4 };

    TGAImage() = default;
    TGAImage(const int w, const int h, const int bpp);
    bool  read_tga_file(const std::string filename);
    bool write_tga_file(const std::string filename, const bool vflip=true, const bool rle=true) const;
    void flip_horizontally();
    void flip_vertically();
    TGAColor get(const int x, const int y) const;
    void set(const int x, const int y, const TGAColor &c);
    int width()  const;
    int height() const;
private:
    bool   load_rle_data(std::ifstream &in);
    bool unload_rle_data(std::ofstream &out) const;

    int w = 0;
    int h = 0;
    std::uint8_t bpp = 0;
    std::vector<std::uint8_t> data = {};
};
