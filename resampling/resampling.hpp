#ifndef RESAMPLING_HPP
#define RESAMPLING_HPP

#include <iostream>
#include <cmath>
#include "interpolation.hpp"


#pragma once

struct resampledImage{
    
    size_t xSize;
    size_t ySize;
    float* resampledData;
};

class Resample{

    public:
        interpolationParams ip;
        const char* interpolationMethod;
        

        Resample(size_t width, size_t height, float srcPixSize, float dstPIxSize, const char* interpolation);
        ~Resample();

        resampledImage resampleImage(float* srcImageData,const char* device);

    private:
        float resamplingFactor;
        bool upscaling=true;
        float* resampledGrid;   


    



};



#endif