#ifndef COMMONS_HPP
#define COMMONS_HPP

#include <iostream>
using namespace std;

#pragma once

//internal image represnetation of this codebase
struct imageData{

    float* data;
    size_t width;
    size_t height;

};

struct interpolationParams{

            size_t originalWidth;
            size_t originalHeight;

            size_t targetWidth;
            size_t targetHeight;

            size_t totalPixels;
            size_t totalResampledPixels;

        };

#endif