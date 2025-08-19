#!/bin/bash
set -e

# Detect the first available CUDA device's compute capability
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -n1)
echo "Detected GPU compute capability: $ARCH"

# Convert compute capability to sm_XY format
SM_ARCH="sm_${ARCH/./}"
echo "Using NVCC arch flag: $SM_ARCH"

# Output directories
BIN_DIR=bin
mkdir -p $BIN_DIR

# Compile mmul.cu to cubin
nvcc -arch=$SM_ARCH -cubin mmul.cu -o $BIN_DIR/mmul.cubin
echo "Generated $BIN_DIR/mmul.cubin"

# Compile genMatrix.cu to fatbin (with curand)
nvcc -arch=$SM_ARCH --fatbin genMatrix.cu -o $BIN_DIR/generate_random_matrix.fatbin -lcudadevrt -lcurand
echo "Generated $BIN_DIR/generate_random_matrix.fatbin"

echo "All kernels compiled successfully!"

