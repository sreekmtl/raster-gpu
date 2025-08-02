#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <string>
using namespace std;

#pragma once

struct interpolationParams{

            size_t originalWidth;
            size_t originalHeight;

            size_t targetWidth;
            size_t targetHeight;

            size_t totalPixels;
            size_t totalResampledPixels;

        };


class Interpolation{

    public:
        float* srcGrid;
        float* targetGrid;
        interpolationParams ip;

        Interpolation(float* srcData, float* targetGrid, interpolationParams interParams, string InterpolationMethod);
        ~Interpolation();

        void nearestNeighbour();

        void bilinear();

        void bicubic();

    private:

};


#endif