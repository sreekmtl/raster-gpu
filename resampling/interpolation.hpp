#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <string>
#include "commons.hpp"
using namespace std;

#pragma once


class Interpolation{

    public:
        float* srcGrid;
        float* targetGrid;
        interpolationParams ip;

        Interpolation(float* srcData, float* targetGrid, interpolationParams interParams, string InterpolationMethod, string device);
        ~Interpolation();

        void nearestNeighbour();

        void bilinear();

        void bicubic();

    private:

};


#endif