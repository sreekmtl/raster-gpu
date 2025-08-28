/**
 * CUDA IMPLEMENTATION OF INTERPOLATION TECHNIQUES
 * RIGHT NOW I AM JUST IMPLEMENTING BICUBIC ON TRIAL BASIS
 */

 #include "interpolation.cuh"
 #include "utils.hpp"
 using namespace std;

 __global__
 void bicubic_kernel(float* paddedData, float* targetData, interpolationParams ip, float scaleX, float scaleY, size_t pad){

    /**
     * SEE OG IMPLEMENTATION IN INTERPOLATION_CPP FOR COMMENTS AND LOGICS
     */

   size_t index= blockIdx.x*blockDim.x+threadIdx.x;
   size_t stride= blockDim.x*gridDim.x;

   // for (size_t i= index; i<numPixels; i+=stride){
   //      result[i]=(band1[i]-band2[i])/(band1[i]+band2[2]);
   //  }

   for (size_t i=index; i<ip.totalResampledPixels; i+=stride){

      size_t y= i/ip.targetWidth;   //height ps
      size_t x= i%ip.targetWidth;   //width pos

      float xSrc= x*scaleX+pad;
      float ySrc= y*scaleY+pad;

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

      float dx= xSrc-x0;
      float dy= ySrc-y0;

      //clamp all above positions to avoid going out of bounds
      if (x1>= paddedImage.width) x1= paddedImage.width-1;
      if (y1>= paddedImage.height) y1= paddedImage.height-1;

      if (x3>= paddedImage.width) x3= paddedImage.width-1;
      if (y3>= paddedImage.height) y3= paddedImage.height-1;

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
        targetData[i] = static_cast<uint16_t>(val);

     }



   }

   

 void bicubic(float* srcData, float* targetData, interpolationParams ip){

    //LETS START IMPLEMETING IT IN CUDA

    size_t pad= 2;
    imageData paddedImage= padImage(srcData, ip.originalWidth, ip.originalHeight, pad);
    size_t paddedPixels= paddedImage.height*paddedImage.width;

    //Now lets move everyting to device memory from host memory
    float * srcData_cu, *targetData_cu, *paddedImage_cu;

    //Allocate device memory
    cudaMalloc(&srcData_cu, ip.totalPixels*(sizeof(float)));
    cudaMalloc(&targetData_cu, ip.totalResampledPixels*(sizeof(float)));
    cudaMalloc(&paddedImage_cu, paddedPixels*(sizeof(float)));

    //copy data from host to device memory
    cudaMemcpy(srcData_cu, srcData, ip.totalPixels*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(paddedImage_cu, paddedImage.data, paddedPixels*sizeof(float), cudaMemcpyHostToDevice);

    //Get the scaling factor
    float scaleX= static_cast<float>(ip.originalWidth)/ip.targetWidth;
    float scaleY= static_cast<float>(ip.originalHeight)/ip.targetHeight;

    int blockSize= 256;
    int numBlocks= (ip.totalPixels+blockSize-1)/blockSize;
    bicubic_kernel<<<numBlocks, blockSize>>>(paddedImage_cu, targetData_cu,ip, scaleX, scaleY, pad);

    cudaDeviceSynchronize();
    cudaMemcpy(targetData, targetData_cu, ip.totalResampledPixels*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(srcData_cu);
    cudaFree(targetData_cu);
    cudaFree(paddedImage_cu);



 }