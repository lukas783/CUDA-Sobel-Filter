/*************************************************************************************************
 * File: imageLoader.cpp
 * Date: 10/16/2017
 *
 * Description: This file is meant to be a file of load/write functions for the original program.
 *      The file includes both a useful typedef to shorten unsigned chars into bytes and a struct
 *      meant to make accessing image data easier.
 *************************************************************************************************/
#include "lodepng.h"
#include <string>

typedef unsigned char byte; // most useful typedef ever

/************************************************************************************************
 * struct imgData(byte*, uint, uint)
 * - a struct to contain all information about our image
 ***********************************************************************************************/
struct imgData {
    imgData(byte* pix = nullptr, unsigned int w = 0, unsigned int h = 0) : pixels(pix), width(w), height(h) {
    };
    byte* pixels;
    unsigned int width;
    unsigned int height;
};

/************************************************************************************************
 * imgData loadImage(char*)
 * - takes in a filename with a png extension and uses LodePNG to decode it, the resulting data
 * - is converted to grayscale and then a structure containing the pixel data, image width, and
 * - height are returned out. Decoding currently uses a C style decoding that uses free and an
 * - array, might consider re-writing to use the newer decode using a C++ vector if I have time.
 *  
 * Inputs: char* filename : the filename of the image to load
 * Outputs:       imgData : a structure returned containing the image data (pixels, width, height)
 * 
 ***********************************************************************************************/
imgData loadImage(char* filename) {
    unsigned int width, height;
    byte* rgb;
    unsigned error = lodepng_decode_file(&rgb, &width, &height, filename, LCT_RGBA, 8);
    if(error) {
        printf("LodePNG had an error during file processing. Exiting program.\n");
        printf("Error code: %u: %s\n", error, lodepng_error_text(error));
        exit(2);
    }
    byte* grayscale = new byte[width*height];
    byte* img = rgb;
    for(int i = 0; i < width*height; ++i) {
        int r = *img++;
        int g = *img++;
        int b = *img++;
        int a = *img++;
        grayscale[i] = 0.3*r + 0.6*g + 0.1*b+0.5;
    }
    free(rgb);
    return imgData(grayscale, width, height);
}

/************************************************************************************************
 * void writeImage(char*, std::string, imgData)
 * - This function takes a filename as a char array, a string of text, and a structure containing
 * - the image's pixel info, width, and height. The function will take the original filename,
 * - remove the .png ending and append text before re-adding the .png extension, then lodepng is
 * - called to encode the pixel data into a png, before the function leaves the pixel data from the
 * - structure is freed as it is not needed anymore.
 * Inputs:        char* filename : the filename of the original image file
 *         std::string appendTxt : the text to append after the original image filename
 *                   imgData img : the structure containing the image's pixel and dimensions
 ***********************************************************************************************/
void writeImage(char* filename, std::string appendTxt, imgData img) {
  std::string newName = filename;
  newName = newName.substr(0, newName.rfind("."));
  newName.append("_").append(appendTxt).append(".png");
  unsigned error = lodepng_encode_file(newName.c_str(), img.pixels, img.width, img.height, LCT_GREY, 8);
    if(error) {
        printf("LodePNG had an error during file writing. Exiting program.\n");
        printf("Error code: %u: %s\n", error, lodepng_error_text(error));
        exit(3);
    }
    delete [] img.pixels;
}
