#!/bin/bash

nvcc -Xcompiler -fPIC -shared libmapalg.cpp compute_indices.cu -o libmapalg.so