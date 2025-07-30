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
        //if its downscaling, then nearest pixels (not strictly) will appear in target's 1 pixel value

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

    /**
     * Bilinear interpolation method
     * 
     * Here we iterate through each pixel in target grid
     * for each of the above pixel, we have to find corresponding replacement pixel from the original image
     * for finding replacement pixel, we take 4 pixels from the original image (within bounds)
     * 4 pixels from bottom-left , bottom right, top right, top left
     * 
     */

     //Here i am adding reflection padding by default
     //since we are considering 4 pixels around pixel in original image, i will be adding 1 pixel padding
     size_t pad= 1;
     size_t padded_width= ip.originalWidth + (2*pad);   //adding padding to top and bottom
     size_t padded_height= ip.originalHeight + (2*pad); //adding padding to left and right
     size_t padded_size= padded_width*padded_height;

     float* paddedImage= new float[padded_size]();
     //fill the inner part 
     size_t k=0;
     for (size_t i=pad; i<padded_height-pad; i++){
        for (size_t j=pad; j<padded_width-pad; j++){
            size_t pos= i * padded_width + j;
            paddedImage[pos]= srcGrid[k];
            k++;
        }
     }

     //now fill the padded area using reflction
     //first we reflect top and bottom
     for (size_t i=0; i<pad; i++){
        for (size_t j=pad; j<padded_width-pad; j++){
            //top
            paddedImage[i*padded_width+j]= paddedImage[(2* pad-i)* padded_width + j];

            //bottom
            paddedImage[(padded_height-1-i)* padded_width + j]= paddedImage[(padded_height - 1 - 2 * pad + i) * padded_width + j];
        }

     }

     //now lets reflect left and right
     //since we already padded top and bottom,
     //while we pad left and right, we can just reflect the top and bottom padded regions too
     //This is easy and straightforward
     for (size_t i=0; i<pad; i++){
        
        for (size_t j=0; j<padded_height; j++){

            //left
            paddedImage[j*padded_width + i]= paddedImage[j * padded_width + (2*pad)- (i+1)];

            //right
            paddedImage[(j+1)*padded_width - i]= paddedImage[(j+1)* padded_width - (2*pad -i)];
        }
        
     }

     

     //Get the scaling factor
     float scaleX= padded_width/ip.targetWidth;
     float scaleY= padded_height/ip.targetHeight;

     for (size_t i=0; i<ip.totalResampledPixels; i++){

        size_t y= i/ip.targetWidth;  //height pos
        size_t x= i%ip.targetWidth;  //width pos


        //now we have to map the location from src image
        float xSrc= x*scaleX;
        float ySrc= y*scaleY;

        //Taking locations of four pixels mentioned in the comment
        //top-left (x_floor, y_floor)
        //top-right (x_ceil, y_floor)
        //bottom-left (x_floor, y_ceil)
        //bottom-right (x_ceil, y_ceil)

        size_t tl= static_cast<size_t>(floor(ySrc)*padded_width + floor(xSrc));
        size_t tr= static_cast<size_t>(floor(ySrc)*padded_width + ceil(xSrc));
        size_t bl= static_cast<size_t>(ceil(ySrc)*padded_width + floor(xSrc));
        size_t br= static_cast<size_t>(ceil(ySrc)* padded_width + ceil(xSrc));

        //we also have to ensure everything is within bounds, if not take position -1 

        //get the actual values from this 4 locations. we are extracting values from the padded image
        float v1= paddedImage[bl];
        float v2= paddedImage[tr];
        float v3= paddedImage[tl];
        float v4= paddedImage[br];

        float q1= v1 * (ceil(xSrc) - xSrc) + v2 * (xSrc - floor(xSrc));
        float q2= v3 * (ceil(xSrc) - xSrc) + v4 * (xSrc - floor(xSrc));

        float q= q1 * (ceil(ySrc) - y) + q2 * (ySrc - floor(ySrc));

        targetGrid[i]= q;
        
     }

     delete [] paddedImage;

}