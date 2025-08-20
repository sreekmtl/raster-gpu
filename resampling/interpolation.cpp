#include <iostream>
#include <cmath>
#include <string>
#include "interpolation.hpp"
#include "utils.hpp"
#include "commons.hpp"
#include <cstdint>

using namespace std;

Interpolation::Interpolation(float* srcData, float* targetData, interpolationParams interParams, string interpolationMethod){

    srcGrid= srcData;
    targetGrid= targetData;
    ip= interParams;

    if (interpolationMethod=="nn"){
        cout<<"Using Nearest Neighbour method"<<endl;
        nearestNeighbour();
        
    }else if (interpolationMethod=="bl"){
        cout<<"Using Bilinear method"<<endl;
        bilinear(); //have to implement bicubic and lanczos
        
    }else if (interpolationMethod=="bc"){
        cout<<"Using Bicubic method"<<endl;
        bicubic();

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

     float scaleX= static_cast<float>(ip.originalWidth)/ip.targetWidth;
     float scaleY= static_cast<float>(ip.originalHeight)/ip.targetHeight;



     for (size_t i=0; i<ip.totalResampledPixels;i++){
        //we have to map pixel now to its col-row position
        //In this current iteration i think since we are querying contigous pix location it will be fast
        //so after crossing each width of image, height will increase by 1

        size_t y= i/ip.targetWidth; //height pos
        size_t x= i%ip.targetWidth; //width pos

        //now we have to map the location from src image
        size_t xSrc= static_cast<size_t>(round(x*scaleX));
        size_t ySrc= static_cast<size_t>(round(y*scaleY));

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
     imageData paddedImage= padImage(srcGrid, ip.originalWidth, ip.originalHeight, 1);

     //Get the scaling factor
     float scaleX= static_cast<float>(ip.originalWidth)/ip.targetWidth;
     float scaleY= static_cast<float>(ip.originalHeight)/ip.targetHeight;

     for (size_t i=0; i<ip.totalResampledPixels; i++){

        size_t y= i/ip.targetWidth;  //height pos
        size_t x= i%ip.targetWidth;  //width pos


        //now we have to map the location from src image
        //here we are not rounding bcos we rounded in nn bcos its `nearest` neighbiur
        float xSrc= x*scaleX+pad;
        float ySrc= y*scaleY+pad;

        //Taking locations of four pixels mentioned in the comment
        //top-left (x_floor, y_floor)
        //top-right (x_ceil, y_floor)
        //bottom-left (x_floor, y_ceil)
        //bottom-right (x_ceil, y_ceil)

        //One thing we have to notice is if my pixel pos is 10.2 and 2.1, then this works correctly as floor and ceil pixel values are different
        //But if my pixel position is 10.0 and 2.0, then problem bcos both floor and ceil gives same value

        //Here eqn is p(unknown)= p1q1+p2q2
        //where q1= (1-x) and q2=x if 1 is distance between two pixels

        float x0= floor(xSrc);
        float y0= floor(ySrc);

        float x1= x0+1;
        float y1= y0+1;

        //dx and dy are calculated to decide the weightage that we have to multipy
        float dx= xSrc-x0;
        float dy= ySrc-y0;

        // Clamp x1 and y1 to avoid going out of bounds
        if (x1 >= paddedImage.width) x1 = paddedImage.width - 1;
        if (y1 >= paddedImage.height) y1 = paddedImage.height - 1;

        if (y0 < 0) y0=0;
        if (x0 < 0) x0=0;

        size_t tl= static_cast<size_t>(y0*paddedImage.width + x0);
        size_t tr= static_cast<size_t>(y0*paddedImage.width + x1);
        size_t bl= static_cast<size_t>(y1*paddedImage.width + x0);
        size_t br= static_cast<size_t>(y1* paddedImage.width + x1);


        float top= paddedImage.data[tl]* (1-dx) + paddedImage.data[tr]*dx;
        float bottom= paddedImage.data[bl]* (1-dx) + paddedImage.data[br]*dx;

        float val= top* (1-dy) + bottom*dy;

        targetGrid[i]= val;
        
     }

     delete [] paddedImage.data;

}

void Interpolation::bicubic(){

    /**
     * Bicubic interpolation method
     * 
     * Here we iterate through each pixel in the target grid
     * For each pixel in the target grid, we have to find the replacement pixel from the srcgrid.
     * For bilinear we took 4 pixels, from srcgrid to interpolate. Here we have to take 16 pixels from src grid.
     * While taking 16 pixels, we have to ensure all these pixels are within bounds
     */

     //Here i am adding reflection padding by default
     //since we are taking 16 pixels, i will be adding 2 pixel padding
     size_t pad= 2;
     imageData paddedImage= padImage(srcGrid, ip.originalWidth, ip.originalHeight, pad);

     //Get the scaling factor
     float scaleX= static_cast<float>(ip.originalWidth)/ip.targetWidth;
     float scaleY= static_cast<float>(ip.originalHeight)/ip.targetHeight;

     for (size_t i=0; i<ip.totalResampledPixels;i++){

        size_t y= i/ip.targetWidth; //height pos
        size_t x= i%ip.targetWidth; //width pos

        //now we have to map the location from src image
        //here we are not rounding similar to nn
        float xSrc= x*scaleX+pad;
        float ySrc= y*scaleY+pad;

        //Taking locations of 16 pixel position
        //Here eqn is p(unknown)= p1q1+p2q2+p3q3+p4q4
        //number coming after vairable have to be considered as power
        //q1= (-x3+2x2-x)/2 |  q2= (3t3-5t2+2)/2 
        //q3= (-3t3+4t2+t)/2 | q4= (t3-t2)/2

        //pixel coordinates of immediate neighbours
        //top-left pixel coordinates
        float x0= floor(xSrc);
        float y0= floor(ySrc);

        //bottom-right pixel coordinates
        float x1= x0+1;
        float y1= y0+1;

        //pixel coordinates of distant neighbours
        //distant top-left neighbours
        float x2= x0-1;
        float y2= y0-1;

        //distant bottom-right neighbours
        float x3= x1+1;
        float y3= y1+1;

        //dx and dy are calculated so we get the x value in the basis function in x direction
        //and y value in the basis function in y direction

        float dx= xSrc-x0;
        float dy= ySrc-y0;

        //clamp all above positions to avoid going out of bounds
        if (x1>= paddedImage.width) x1= paddedImage.width-1;
        if (y1>= paddedImage.height) y1= paddedImage.height-1;

        if (x3>= paddedImage.width) x3= paddedImage.width-1;
        if (y3>= paddedImage.height) y3= paddedImage.height-1;

        // if (x2 < 0) x2=0;
        // if (y2 < 0) y2=0;

        //now we have to find the positions of 16 pixel position
        //for that we are going to loop

        size_t p1= static_cast<size_t>(y2*paddedImage.width+x2);
        size_t p2= static_cast<size_t>(y2*paddedImage.width+x0);
        size_t p3= static_cast<size_t>(y2*paddedImage.width+x1);
        size_t p4= static_cast<size_t>(y2*paddedImage.width+x3);

        size_t p5= static_cast<size_t>(y0*paddedImage.width+x2);
        size_t p6= static_cast<size_t>(y0*paddedImage.width+x0);
        size_t p7= static_cast<size_t>(y0*paddedImage.width+x1);
        size_t p8= static_cast<size_t>(y0*paddedImage.width+x3);

        size_t p9= static_cast<size_t>(y1*paddedImage.width+x2);
        size_t p10= static_cast<size_t>(y1*paddedImage.width+x0);
        size_t p11= static_cast<size_t>(y1*paddedImage.width+x1);
        size_t p12= static_cast<size_t>(y1*paddedImage.width+x3);

        size_t p13= static_cast<size_t>(y3*paddedImage.width+x2);
        size_t p14= static_cast<size_t>(y3*paddedImage.width+x0);
        size_t p15= static_cast<size_t>(y3*paddedImage.width+x1);
        size_t p16= static_cast<size_t>(y3*paddedImage.width+x3);
        
        //hotizontal weights
        float dx2= dx*dx;
        float dx3= dx2*dx;
        
        float h1 = (-dx3 + 2*dx2 - dx) / 2.0f;
        float h2 = (3*dx3 - 5*dx2 + 2) / 2.0f;
        float h3 = (-3*dx3 + 4*dx2 + dx) / 2.0f;
        float h4 = (dx3 - dx2) / 2.0f;

        float row1= static_cast<float>(paddedImage.data[p1])*h1 + static_cast<float>(paddedImage.data[p2])*h2 + static_cast<float>(paddedImage.data[p3])*h3 + static_cast<float>(paddedImage.data[p4])*h4;
        float row2= static_cast<float>(paddedImage.data[p5])*h1 + static_cast<float>(paddedImage.data[p6])*h2 + static_cast<float>(paddedImage.data[p7])*h3 + static_cast<float>(paddedImage.data[p8])*h4;
        float row3= static_cast<float>(paddedImage.data[p9])*h1 + static_cast<float>(paddedImage.data[p10])*h2 + static_cast<float>(paddedImage.data[p11])*h3 + static_cast<float>(paddedImage.data[p12])*h4;
        float row4= static_cast<float>(paddedImage.data[p13])*h1 + static_cast<float>(paddedImage.data[p14])*h2 + static_cast<float>(paddedImage.data[p15])*h3 + static_cast<float>(paddedImage.data[p16])*h4;

        //vertical weights
        float dy2= dy*dy;
        float dy3= dy2*dy;
        float v1= (-dy3+ 2*dy2- dy)/2.0f;
        float v2= (3*dy3- 5*dy2+2)/2.0f;
        float v3= (-3*dy3+ 4*dy2 + dy)/2.0f;
        float v4= (dy3-dy2)/2.0f;

        float val = v1*row1 + v2*row2 + v3*row3 + v4*row4;
        val = std::min(std::max(val, 0.0f), 65535.0f);
        targetGrid[i] = static_cast<uint16_t>(val);


     }

     delete [] paddedImage.data;
     
}