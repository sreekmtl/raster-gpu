#ifndef COMPUTE_INDICES_CUH
#define COMPUTE_INDICES_CUH

#include <iostream>
#include <cuda_runtime.h>

#pragma once

class ComputeIndices{

    public:
        ComputeIndices();
        ~ComputeIndices();

        void NormalizedIndices(float* band1, float* band2, float* h_result, size_t numPixels);
};

#endif