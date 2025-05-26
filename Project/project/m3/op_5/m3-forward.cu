#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#define TILE_WIDTH 16

__global__ void matmul_conv_fused_fp16(const float *mask, const float *input, float *output,
                                      int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    // Shared memory tiles
    __shared__ __half2 tile_mask[TILE_WIDTH][TILE_WIDTH]; // A
    __shared__ __half2 tile_input[TILE_WIDTH][TILE_WIDTH]; // B

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

    int row = by * TILE_WIDTH + ty; // represents m
    int col = bx * TILE_WIDTH + tx;

    int b = col / image_size;
    int hw = col % image_size;
    int h = hw / Width_out;
    int w = hw % Width_out;

    __half2 val = __floats2half2_rn(0.0f, 0.0f); // Initialize __half2 to (0, 0)

    for (int tileId = 0; tileId < (numAColumns - 1) / (TILE_WIDTH * 2) + 1; tileId++) {
        size_t tiledIdx = tileId * TILE_WIDTH * 2; // Each __half2 handles 2 values
        // Load tile_mask
        if (row < numARows && tiledIdx + tx * 2 < numAColumns) {
            size_t idx = tiledIdx + tx * 2;
            int c = idx / KK;
            int pq = idx % KK;
            int p = pq / K;
            int q = pq % K;

            // Load two consecutive values into __half2
            float val0, val1;
            if (idx < numAColumns) val0 = mask_4d(row, c, p, q);
            else val0 = 0.0f;

            if (idx + 1 < numAColumns) val1 = mask_4d(row, c + (pq + 1) / KK, (pq + 1) % KK / K, (pq + 1) % K);
            else val1 = 0.0f;

            tile_mask[ty][tx] = __floats2half2_rn(val0, val1);
        } else {
            tile_mask[ty][tx] = __floats2half2_rn(0.0f, 0.0f);
        }

        // Load tile_input
        if (col < numBColumns && tiledIdx + ty * 2 < numBRows) {
            size_t idx = tiledIdx + ty * 2;
            int c = idx / KK;
            int pq = idx % KK;
            int p = pq / K;
            int q = pq % K;

            // Load two consecutive values into __half2
            float val0, val1;
            if (idx < numAColumns) val0 = in_4d(b, c, h + p, w + q);
            else val0 = 0.0f;
            if (idx + 1 < numAColumns) val1 = in_4d(b, c + (pq + 1) / KK, h + ((pq + 1) % KK) / K, w + ((pq + 1) % K));
            else val1 = 0.0f;

            tile_input[ty][tx] = __floats2half2_rn(val0, val1);
        } else {
            tile_input[ty][tx] = __floats2half2_rn(0.0f, 0.0f);
        }
        __syncthreads();

        // Compute with __half2 operations
        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val = __hadd2(val, __hmul2(tile_mask[ty][i], tile_input[i][tx]));
            }
        }
        __syncthreads();
    }

    // Write output
    if (row < numCRows && col < numCColumns) {
        // Sum the two __half values in val to get the final result
        float result = __half2float(val.x) + __half2float(val.y);
        output[b * Map_out * image_size + row * image_size + hw] = result;
    }
}
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);  
    printf("Task op_5 with __half2\n");
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

    matmul_conv_fused_fp16<<<DimGrid, DimBlock>>>(
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