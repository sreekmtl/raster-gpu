#include <iostream>
#include "resampling.hpp"

using namespace std;

Resample::Resample(size_t width, size_t height, float srcPixSize, float dstPixSize, const char* interpolation ){

    /**
     * Based on given details in constructor, we have to create a destination grid for interpolation
     * Assuimg all pixel are square
     */

    interpolationMethod= interpolation;
    size_t totalPixels= width*height;


    if (dstPixSize>srcPixSize){
        upscaling=false ;

    }else if (dstPixSize==srcPixSize){
        //dont have to do anything we have to exit
    }

    resamplingFactor= static_cast<float>(dstPixSize/srcPixSize);

    //Deriving target width and height
    size_t resampledWidth= static_cast<size_t>(ceil(width*resamplingFactor));
    size_t resampledHeight= static_cast<size_t>(ceil(height*resamplingFactor));
    size_t totalResampledPixels= resampledWidth*resampledHeight;

    //Now lets create the target grid
    resampledGrid= new float[totalResampledPixels]();

    //set interpolation params
    
    ip.originalWidth=  width;
    ip.originalWidth= height;
    ip.targetWidth= resampledWidth;
    ip.targetHeight= resampledHeight;
    ip.totalPixels= totalPixels;
    ip.totalResampledPixels= totalResampledPixels;

}

Resample::~Resample(){

}

void Resample::resampleImage(float* srcImageData, float* resampledImageData){

    /**
     * We have the srcimage data and details, targetgrid and details
     * and the interpolation method to use
     * AS OF NOW ONLY IMPLEMENTING NEAREST NEIGHBOUR
     */

    Interpolation interpolator(srcImageData, resampledImageData, ip, interpolationMethod);
    interpolator.nearestNeighbour();

    
}
