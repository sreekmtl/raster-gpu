#include <iostream>
#include <assert.h>
#include <random>
#include <vector>
#include "nvcomp_12/nvcomp.hpp"
#include "nvcomp_12/nvcomp/lz4.hpp"
#include "nvcomp_12/nvcomp/nvcompManagerFactory.hpp"
#include "nvcomp_12/nvcomp/lz4.h"

using namespace std;
using namespace nvcomp;



/**
 * In this example, we compress and decompress the data
 */


void execute_example(char* input_data, const size_t in_bytes){

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t* host_uncompressed_bytes;
    const size_t chunk_size= 65536;
    const size_t batch_size= (in_bytes + chunk_size -1)/chunk_size;

    char* device_input_data;
    cudaMalloc(&device_input_data, in_bytes);
    cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream);

    cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t)*batch_size);

    for(size_t i=0; i<batch_size; i++){
        if(i+1 <batch_size){
            host_uncompressed_bytes[i]= chunk_size;
        }else{
            host_uncompressed_bytes[i]= in_bytes-(chunk_size*i);
        }
    }

    void** host_uncompressed_ptrs;
    cudaMallocHost(&host_uncompressed_ptrs, sizeof(size_t)*batch_size);
    for(size_t ix_chunk=0; ix_chunk< batch_size; ix_chunk++){
        host_uncompressed_ptrs[ix_chunk]= device_input_data+chunk_size*ix_chunk;
    }

    size_t* device_uncompressed_bytes;
    void** device_uncompressed_ptrs;

    cudaMalloc(&device_uncompressed_bytes, sizeof(size_t)*batch_size);
    cudaMalloc(&device_uncompressed_ptrs, sizeof(size_t)*batch_size);

    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t)*batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(size_t)*batch_size, cudaMemcpyHostToDevice, stream);

    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void * device_temp_ptr;
    cudaMalloc(&device_temp_ptr, temp_bytes);

    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    void** host_compressed_ptrs;
    cudaMallocHost(&host_compressed_ptrs, sizeof(size_t)*batch_size);
    for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        cudaMalloc(&host_compressed_ptrs[ix_chunk], max_out_bytes);
    }
  
    void** device_compressed_ptrs;
    cudaMalloc(&device_compressed_ptrs, sizeof(size_t) * batch_size);
    cudaMemcpyAsync(
        device_compressed_ptrs, host_compressed_ptrs, 
        sizeof(size_t) * batch_size,cudaMemcpyHostToDevice, stream);
  
    
    size_t * device_compressed_bytes;
    cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);

    nvcompStatus_t comp_res= nvcompBatchedLZ4CompressAsync(
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        chunk_size,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        device_compressed_ptrs,
        device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        stream

    );

    cudaStreamSynchronize(stream);


    if (comp_res != nvcompSuccess){
        cerr<<"Failed compression"<<endl;
        assert(comp_res==nvcompSuccess);
    }

    //TO SEE COMPRESSED DATA

    // void* h_compressed_data[batch_size];
    // size_t h_compressed_sizes[batch_size];

    // cudaMemcpy(h_compressed_sizes, device_compressed_bytes, sizeof(size_t)*batch_size, cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < batch_size; ++i) {
    //     h_compressed_data[i] = malloc(h_compressed_sizes[i]);
    //     void* d_ptr;
    //     cudaMemcpy(&d_ptr, &device_compressed_ptrs[i], sizeof(void*), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_compressed_data[i], d_ptr, h_compressed_sizes[i], cudaMemcpyDeviceToHost);
    // }

    // for (size_t k = 0; k < batch_size; k++) {
    //     // Cast the void pointer to char* to handle byte-by-byte printing
    //     char* data = static_cast<char*>(h_compressed_data[k]);
    
    //     // Print the compressed data byte by byte
    //     for (size_t l = 0; l < h_compressed_sizes[k]; l++) {
    //         cout << static_cast<int>(data[l]) << " ";  // Print each byte as an integer (in decimal)
    //     }
    //     cout << endl;
    // }

    // After cudaStreamSynchronize(stream);
    size_t* host_compressed_bytes = (size_t*)malloc(sizeof(size_t) * batch_size);
    cudaMemcpy(host_compressed_bytes, device_compressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost);

    // Print the compressed sizes
    for (size_t i = 0; i < batch_size; ++i) {
        std::cout << "Chunk " << i << " compressed size: " << host_compressed_bytes[i] << " bytes" << std::endl;
    }


    //TO UNCOMPRESS

    nvcompBatchedLZ4GetDecompressSizeAsync(
        device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, batch_size, stream
    );

    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes);
    void *device_decomp_temp;
    cudaMalloc(&device_decomp_temp, decomp_temp_bytes);

    nvcompStatus_t* device_statuses;
    cudaMalloc(&device_statuses, sizeof(nvcompStatus_t)*batch_size);

    size_t* device_actual_uncompressed_bytes;
    cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t)*batch_size);

    nvcompStatus_t decomp_res= nvcompBatchedLZ4DecompressAsync(
        device_compressed_ptrs, 
        device_compressed_bytes, 
        device_uncompressed_bytes, 
        device_actual_uncompressed_bytes, 
        batch_size,
        device_decomp_temp, 
        decomp_temp_bytes, 
        device_uncompressed_ptrs, 
        device_statuses, 
        stream
    );

    if (decomp_res != nvcompSuccess){
        cerr<<"Failed decompress"<<endl;
        assert(decomp_res == nvcompSuccess);
    }

    cudaStreamSynchronize(stream);

    // === Cleanup ===

    cudaFree(device_input_data);
    cudaFree(device_uncompressed_bytes);
    cudaFree(device_uncompressed_ptrs);
    cudaFree(device_temp_ptr);
    cudaFree(device_compressed_ptrs);
    cudaFree(device_compressed_bytes);
    cudaFree(device_decomp_temp);
    cudaFree(device_statuses);
    cudaFree(device_actual_uncompressed_bytes);

    for (size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
        cudaFree(host_compressed_ptrs[ix_chunk]);
    }

    cudaFreeHost(host_uncompressed_bytes);
    cudaFreeHost(host_uncompressed_ptrs);
    cudaFreeHost(host_compressed_ptrs);

    free(host_compressed_bytes);

    cudaStreamDestroy(stream);

    

    
}

int main(){

    //Initializing random array of chars
    const size_t in_bytes= 1000000;
    char* uncompressed_data;

    cudaMallocHost(&uncompressed_data, in_bytes);
    mt19937 random_gen(42);

    uniform_int_distribution<short> uniform_dist(0,255);
    for (size_t ix=0; ix < in_bytes; ix++){
        uncompressed_data[ix]= static_cast<char>(uniform_dist(random_gen));
    }

    
    execute_example(uncompressed_data, in_bytes);
    return 0;



    return 0;
}