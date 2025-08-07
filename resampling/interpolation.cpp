#include <iostream>
#include <cmath>
#include <string>
#include "interpolation.hpp"

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

        float x0= floor(xSrc);
        float y0= floor(ySrc);

        float x1= x0+1;
        float y1= y0+1;

        //dx and dy are calculated to decide the weightage that we have to multipy
        float dx= xSrc-x0;
        float dy= ySrc-y0;

        // Clamp x1 and y1 to avoid going out of bounds
        if (x1 >= padded_width) x1 = padded_width - 1;
        if (y1 >= padded_height) y1 = padded_height - 1;

        size_t tl= static_cast<size_t>(y0*padded_width + x0);
        size_t tr= static_cast<size_t>(y0*padded_width + x1);
        size_t bl= static_cast<size_t>(y1*padded_width + x0);
        size_t br= static_cast<size_t>(y1* padded_width + x1);


        float top= paddedImage[tl]* (1-dx) + paddedImage[tr]*dx;
        float bottom= paddedImage[bl]* (1-dx) + paddedImage[br]*dx;

        float val= top* (1-dy) + bottom*dy;

        targetGrid[i]= val;
        
     }

     delete [] paddedImage;

}