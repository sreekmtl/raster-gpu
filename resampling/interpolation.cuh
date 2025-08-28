#ifndef INTERPOLATION_CUH
#define INTERPOLATION_CUH

#include <iostream>
#include <cuda_runtime.h>
#include "commons.hpp"

#pragma once

void bicubic(float* srcData, float* targetData, interpolationParams params);


#endif