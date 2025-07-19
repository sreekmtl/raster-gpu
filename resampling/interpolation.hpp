#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#pragma once
class Interpolation{

    public:
        float* srcGrid;
        float* targetGrid;
        const char* interpolationMehtod;
        interpolationParams ip;

        Interpolation(float* srcData, float* targetGrid, interpolationParams interParams, const char* InterpolationMethod);
        ~Interpolation();

        void nearestNeighbour();

        void bilinear();

        void bicubic();

    private:

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