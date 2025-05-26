#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_runtime.h>
#include <mma.h>
#include <stdlib.h>
#include <string.h>
using namespace nvcuda;

#define TILE_WIDTH 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

__global__ void matmul_conv_fused_tensor_cores(const float *mask, const float *input, float *output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    

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
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0] // mask(m,c,p,q)
    
    int block_row = blockIdx.y * WMMA_M;
    int block_col = blockIdx.x * WMMA_N;

    // Shared memory tiles
    
    extern __shared__ float shared[];
    float* tile_mask = shared;                                // 16x8 = 128 floats
    float* tile_input = tile_mask + WMMA_M * WMMA_K;                        // 8*16 = 128 floats
    float* tile_C = tile_input + WMMA_K * WMMA_N;                        // 16*16 = 256 floats
    
    // ==== Tensor Core multiply ====
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int dx_a, dy_a, dx_b, dy_b;
    size_t row_a, col_a, row_b, col_b;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        for(int t=0;t<2;t++){
            // ==== Load A and B into shared memory ====
            size_t tiledIdx = tileId * TILE_WIDTH+WMMA_K*t; //tile_A*tile_B = 16*16 matrix
            dx_a = tx%8;
            dy_a = tx/8; // 0, 1, 2, 3, 4
            dx_b = tx%16;
            dy_b = tx/16; // 0, 1
            for(int i=0;i<4;i++){
                row_a = block_row + dy_a*4+i;
                col_a = tiledIdx + dx_a;
                row_b = tiledIdx + dy_b*4+i;
                col_b = block_col + dx_b;
                
                int b = col_b/image_size; // for input (matrix b)
                int hw = col_b%image_size; // x in the unroll c
                int h = hw/Width_out;
                int w = hw%Width_out;

                if (row_a < numARows && col_a < numAColumns) {
                    size_t idx = col_a;
                    int c = idx / KK;
                    int pq = idx % KK;
                    int p = pq / K ;
                    int q = pq % K ;

                    tile_mask[(dy_a*4+i)*WMMA_K+dx_a] = mask_4d(row_a,c,p,q);
                } else {
                    tile_mask[(dy_a*4+i)*WMMA_K+dx_a] = 0.0f;
                }
                if (col_b < numBColumns && row_b < numBRows) {
                    size_t idx = row_b;
                    int c = idx / KK;
                    int pq = idx % KK;
                    int p = pq/K;
                    int q = pq%K;

                    tile_input[(dy_b*4+i)*WMMA_N+dx_b] = in_4d(b,c,h+p,w+q);
                } else {
                    tile_input[(dy_b*4+i)*WMMA_N+dx_b] = 0.0f;
                }    
            }
            __syncthreads();

            // tensor core matmul
            wmma::load_matrix_sync(a_frag, tile_mask, WMMA_K); // stride is full shared row size
            wmma::load_matrix_sync(b_frag, tile_input, WMMA_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
        
    }
    wmma::store_matrix_sync(tile_C, c_frag, WMMA_N, wmma::mem_row_major);
    __syncthreads();

    int dx = tx%16;
    int dy = tx/16;
    // 16*16/32 = 8
    // each thread will need to store 8 elements to C
    for(int i=0;i<8;i++){
        int row = block_row+dy*8+i;// represents m
        int col = block_col+dx;
        int b = col/image_size;
        int hw = col%image_size; // x in the unroll c
        int h = hw/Width_out;
        int w = hw%Width_out;
        if (row < numCRows && col < numCColumns) {
            output[b * Map_out * image_size + row * image_size + hw] = tile_C[(dy*8+i)*TILE_WIDTH+dx];
        }
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
    printf("Task: req_1\n");
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
    dim3 DimBlock(32, 1, 1);
    size_t shmem_size = (WMMA_M * WMMA_K + WMMA_K * WMMA_N + WMMA_M * WMMA_N) * sizeof(float);

    matmul_conv_fused_tensor_cores<<<DimGrid, DimBlock, shmem_size>>>(
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