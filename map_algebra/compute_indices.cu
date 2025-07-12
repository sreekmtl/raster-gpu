
#include "compute_indices.cuh"

using namespace std;


__global__
void normalizedindices(size_t numPixels, float* band1, float* band2, float* result){

    size_t index= blockIdx.x*blockDim.x+threadIdx.x;
    size_t stride= blockDim.x*gridDim.x;

    for (size_t i= index; i<numPixels; i+=stride){
        result[i]=(band1[i]-band2[i])/(band1[i]+band2[2]);
    }
}




ComputeIndices::ComputeIndices(){

}

ComputeIndices::~ComputeIndices(){

}

void ComputeIndices::NormalizedIndices(float* h_band1, float* h_band2, float* h_result, size_t numPixels){

    //Initialize memory on device and then we have to copy data from host memory to device memory
    float *d_band1, *d_band2, *d_result;

    //Allocate device memory
    cudaMalloc(&d_band1, numPixels*(sizeof(float)));
    cudaMalloc(&d_band2, numPixels*(sizeof(float)));
    cudaMalloc(&d_result, numPixels*(sizeof(float)));

    //Now lets copy data from host memory to device memory
    cudaMemcpy(d_band1, h_band1, numPixels*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_band2, h_band2, numPixels*sizeof(float), cudaMemcpyHostToDevice);

    int blockSize= 256;
    int numBlocks= (numPixels+blockSize-1)/blockSize;
    normalizedindices<<<numBlocks, blockSize>>>(numPixels, d_band1, d_band2, d_result);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, numPixels*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_band1);
    cudaFree(d_band2);
    cudaFree(d_result);



}



