#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cassert>

#define task "op_0"
#define TILE_WIDTH 16
#define MAX_CONST_MASK_SIZE 4000 
__constant__ float const_mask[MAX_CONST_MASK_SIZE];

__global__ void matmul_conv_fused_with_constant_mask(const float *input, float *output,
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

    dim3 DimGrid(ceil(Map_out / TILE_WIDTH), ceil((Batch * H_out * W_out) / TILE_WIDTH), 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    */
    // Shared memory tiles
    
    __shared__ float tile_mask[TILE_WIDTH][TILE_WIDTH]; // A
    __shared__ float tile_input[TILE_WIDTH][TILE_WIDTH];// B
    // compute C = A*B

    const int Height_out = Height-K+1;
    const int Width_out = Width-K+1;
    const int KK = K*K;
    const int image_size = Height_out*Width_out;

    const int numARows = Map_out;
    const int numAColumns = K*K*Channel;
    const int numBRows = numAColumns;
    const int numBColumns = Batch*image_size;
    

    const int numCRows = numARows;
    const int numCColumns = numBColumns;
    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0] //input(b,c,h+p,w+q)
    #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // mask(m,c,p,q)
    
    int row = by * TILE_WIDTH + ty; // represents m
    int col = bx * TILE_WIDTH + tx;

    int b = col/image_size;
    int hw = col%image_size; // x in the unroll c
    int h = hw/Width_out;
    int w = hw%Width_out;
    
    float val = 0;
    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        size_t tiledIdx = tileId * TILE_WIDTH;
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {

            size_t idx = tiledIdx + tx;
            int c = idx / KK;
            int pq = idx % KK;
            int p = pq / K ;
            int q = pq % K ;

            tile_mask[ty][tx] = mask_4d(row,c,p,q);
        } else {
            tile_mask[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            size_t idx = tiledIdx + ty;
            int c = idx / KK;
            int pq = idx % KK;
            int p = pq/K;
            int q = pq%K;

            tile_input[ty][tx] = in_4d(b,c,h+p,w+q);
        } else {
            tile_input[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tile_mask[ty][i] * tile_input[i][tx];
            }
        }
        __syncthreads();
    }
    if (row < numCRows && col < numCColumns) {
        output[b * Map_out * image_size + row * image_size + hw] = val;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);  
    std::cout << "mask size (bytes): " << mask_size << std::endl;


    cudaMalloc(device_input_ptr, input_size);
    cudaMalloc(device_output_ptr, output_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    std::cout<<"Task: "<<task<<std::endl;
    cudaError_t err = cudaMemcpyToSymbol(const_mask, host_mask, mask_size, 0,cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol error: " << cudaGetErrorString(err) << std::endl;
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

    matmul_conv_fused_with_constant_mask<<<DimGrid, DimBlock>>>(
     device_input, device_output,
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