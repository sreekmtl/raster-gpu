#include <iostream>
#include "compute_indices.cuh"

using namespace std;

extern "C"{

    void normalized_indices(float* band1, float* band2, float* result, size_t numPixels);
}


void normalized_indices(float* band1, float* band2, float* result, size_t numPixels){

    ComputeIndices ci;
    ci.NormalizedIndices(band1, band2, result, numPixels);

}