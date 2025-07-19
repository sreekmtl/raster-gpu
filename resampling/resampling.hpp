#ifndef RESAMPLING_HPP
#define RESAMPLING_HPP

#include <iostream>
#include <cmath>
#include "interpolation.hpp"


#pragma once

class Resample{

    public:
        interpolationParams ip;
        const char* interpolationMethod;
        

        Resample(size_t width, size_t height, float srcPixSize, float dstPIxSize, const char* interpolation);
        ~Resample();

        void resampleImage(float* srcImageData, float* resampledImageData);

    private:
        float resamplingFactor;
        bool upscaling=true;
        float* resampledGrid;   


    



};

#endif