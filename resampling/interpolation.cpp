#include <iostream>
#include <cmath>
#include "interpolation.hpp"

using namespace std;

Interpolation::Interpolation(float* srcData, float* targetGrid, interpolationParams interParams, const char* interpolationMethod){

    srcData= srcData;
    targetGrid= targetGrid;
    ip= interParams;
    
    
    interpolationMethod= interpolationMethod;

    if (interpolationMethod=="nn"){
        nearestNeighbour();
    }else if (interpolationMethod=="bl"){
        bilinear(); //have to implement bicubic and lanczos
    }else{
        cerr<<"Invalid interpolation method"<<endl;
    }

    


}

Interpolation::~Interpolation(){

}

void Interpolation::nearestNeighbour(){

    /**
     * Nearest Neighbour interpolation method
     * 
     * First we have to calculate the scaleX and scaleY bcos this is a scaling problem. 
     * Based on upscaling/downscaling, we get a scale and based on that scale and the distance to neighbouring pixel in ....
     * 
     * So secondly we have to iterate through each pixel in the target grid 
     * so for each pixel in the target grid we have to find a pixel value.
     */

     float scaleX= ip.originalWidth/ip.targetWidth;
     float scaleY= ip.originalHeight/ip.targetHeight;


     for (size_t i=0; i<ip.totalResampledPixels;i++){

        //we have to map pixel now to its col-row position
        //In this current iteration i think since we are querying contigous pix location it will be fast
        //so after crossing each width of image, height will increase by 1

        size_t y= i/ip.targetWidth; //height pos
        size_t x= i%ip.targetWidth; //width pos

        //now we have to map the location from src image

        size_t xSrc= round(x*scaleX);
        size_t ySrc= round(y*scaleY);

        //so in upscaling, we will be having more pixels in targetimage
        //so if resampling factor is 2, then 1 pixel value will appear in 4 nearby pixels in target grid
        //if its downscaling, then nearest of 4 pixels (not strictly) will appear in target's 1 pixel value

        //so i think here near means nearest in terms of euclidean distance
        //euclidean distance between target pixel's coordinates and src coordinates pixels are measured ?
        //we also have to check that the mapped pixel in src image is within the bounds


        //checking bounds, if rounded positions are above with or height, keep it within bounds
        if (xSrc>=ip.originalWidth){
            xSrc= ip.originalWidth-1;
        }

        if (ySrc>=ip.originalHeight){
            ySrc= ip.originalHeight-1;
        }

        size_t srcPixelPosition= static_cast<size_t>((ySrc * ip.originalWidth) + xSrc); 
        targetGrid[i]= srcGrid[srcPixelPosition];
        

     }

}

void Interpolation::bilinear(){

}