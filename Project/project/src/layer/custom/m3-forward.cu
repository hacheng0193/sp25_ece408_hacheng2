#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <stdlib.h>
#include <string.h>    
#include <cuda_fp16.h> // fp16 optimization
using namespace nvcuda;

#define TILE_WIDTH 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void matmul_conv_fused_final(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    extern __shared__ float shared[];
    // Shared memory tiles
    __half* tile_mask = (__half*)shared;  // FP16 Mask Tile
    __half* tile_input = (__half*)tile_mask + WMMA_M * WMMA_K;  // FP16 Input Tile
    float* tile_C = (float*)tile_input + WMMA_K * WMMA_N;  // Float C Tile

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int KK = K * K;
    const int image_size = Height_out * Width_out;

    const int numARows = Map_out;
    const int numAColumns = K * K * Channel;
    const int numBRows = numAColumns;
    const int numBColumns = Batch * image_size;

    const int numCRows = numARows;
    const int numCColumns = numBColumns;
    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    
    int block_row = blockIdx.y * WMMA_M;
    int block_col = blockIdx.x * WMMA_N;

    // ==== Tensor Core multiply ====

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int row = by * TILE_WIDTH + ty;  // represents m
    int col = bx * TILE_WIDTH + tx;

    int b = col / image_size;
    int hw = col % image_size;  // x in the unroll c
    int h = hw / Width_out;
    int w = hw % Width_out;
    
    // === Full Size tile ===
    for (int tileId = 0; tileId < (numAColumns / WMMA_K); tileId++) {
        size_t tiledIdx = tileId * WMMA_K;

        // Load Matrix A
        if (row < numARows) {
            int idx = tiledIdx + tx;
            int c = idx / KK;
            int pq = idx % KK;
            int p = pq / K;
            int q = pq % K;
            tile_mask[ty * WMMA_K + tx] = __float2half(mask_4d(row, c, p, q));
        }

        // Load Matrix B
        if (col < numBColumns) {
            int idx = tiledIdx + ty;
            int c = idx / KK;
            int pq = idx % KK;
            int p = pq / K;
            int q = pq % K;
            tile_input[ty * WMMA_K + tx] = __float2half(in_4d(b, c, h + p, w + q));
        }

        __syncthreads();

        // Tensor Core GEMM
        if (ty < 2) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, tile_mask, WMMA_K);
            wmma::load_matrix_sync(b_frag, tile_input, WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }

    // === last tile padding zero ===
    int last_tile_start = (numAColumns / WMMA_K) * WMMA_K;
    if (last_tile_start < numAColumns) {
        int tileId = numAColumns / WMMA_K;  // last tile

        size_t tiledIdx = tileId * WMMA_K;

        // Load Matrix A
        if (row < numARows) {
            int idx = tiledIdx + tx;
            if (idx < numAColumns) {
                int c = idx / KK;
                int pq = idx % KK;
                int p = pq / K;
                int q = pq % K;
                tile_mask[ty * WMMA_K + tx] = __float2half(mask_4d(row, c, p, q));
            } else {
                tile_mask[ty * WMMA_K + tx] = __float2half(0.0f);
            }
        }

        // Load Matrix B
        if (col < numBColumns) {
            int idx = tiledIdx + ty;
            if (idx < numBRows) {
                int c = idx / KK;
                int pq = idx % KK;
                int p = pq / K;
                int q = pq % K;
                tile_input[ty * WMMA_K + tx] = __float2half(in_4d(b, c, h + p, w + q));
            } else {
                tile_input[ty * WMMA_K + tx] = __float2half(0.0f);
            }
        }

        __syncthreads();

        // Tensor Core GEMM
        if (ty < 2) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, tile_mask, WMMA_K);
            wmma::load_matrix_sync(b_frag, tile_input, WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        __syncthreads();
    }
    if (ty < 2) wmma::store_matrix_sync(tile_C, c_frag, WMMA_N, wmma::mem_row_major);

    __syncthreads();

    if (row < numCRows && col < numCColumns) {
        output[b * Map_out * image_size + row * image_size + hw] = tile_C[ty * WMMA_N + tx];
    }
}




__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);  

    cudaMalloc(device_input_ptr, input_size);
    cudaMalloc(device_output_ptr, output_size);
    cudaMalloc(device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
    
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    dim3 DimGrid((Width_unrolled - 1) / TILE_WIDTH + 1,
                         (Map_out - 1) / TILE_WIDTH + 1, 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    //size_t shmem_size = (2 * WMMA_M * WMMA_K + 2 * WMMA_K * WMMA_N + WMMA_M * WMMA_N) * sizeof(float);
    size_t shmem_size = (WMMA_M * WMMA_K + WMMA_K * WMMA_N + WMMA_M * WMMA_N) * sizeof(float);
    matmul_conv_fused_final<<<DimGrid, DimBlock, shmem_size>>>(
    device_mask, device_input, device_output,
    Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int input_size = Batch*Channel*Height*Width*sizeof(float);
    int output_size = Batch*Map_out*(Height - K + 1)*(Width - K + 1)*sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}