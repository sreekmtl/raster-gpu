#include "utils.hpp"

float* padImage(float* imageData, size_t width, size_t height, size_t pad){

    size_t padded_width= width + (2*pad);   //adding padding to top and bottom
    size_t padded_height= height + (2*pad); //adding padding to left and right
    size_t padded_size= padded_width*padded_height;

    float* paddedImage= new float[padded_size]();
    //fill the inner part 
    size_t k=0;
    for (size_t i=pad; i<padded_height-pad; i++){
    for (size_t j=pad; j<padded_width-pad; j++){
        size_t pos= i * padded_width + j;
        paddedImage[pos]= imageData[k];
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

    return paddedImage;

}