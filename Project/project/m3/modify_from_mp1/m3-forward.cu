#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define tile_width 16

#define MAX_CONST_MASK_SIZE 4000 
__constant__ float const_mask[MAX_CONST_MASK_SIZE];

__global__ void matmul_conv_fused_final(const float *__restrict__ input,
                                  float *__restrict__ output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K, int H_grid, int W_grid, int image_size) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) const_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int m = blockIdx.y;

    int b = blockIdx.x / image_size;

    int hw  = blockIdx.x % image_size;
    int h = (hw / W_grid) * tile_width + threadIdx.y;
    int w = (hw % W_grid) * tile_width + threadIdx.x;

    float sum=0;
    if(h<Height_out && w<Width_out && b<Batch){
        for(int c=0;c<Channel;c++){// sum over all input feature maps
            for(int p=0;p<K;p++){// KxK filter, K=7
                sum += in_4d(b,c,h+p,w)*mask_4d(m,c,p,0);
                sum += in_4d(b,c,h+p,w+1)*mask_4d(m,c,p,1);
                sum += in_4d(b,c,h+p,w+2)*mask_4d(m,c,p,2);
                sum += in_4d(b,c,h+p,w+3)*mask_4d(m,c,p,3);
                sum += in_4d(b,c,h+p,w+4)*mask_4d(m,c,p,4);
                sum += in_4d(b,c,h+p,w+5)*mask_4d(m,c,p,5);
                sum += in_4d(b,c,h+p,w+6)*mask_4d(m,c,p,6);
            }
        
        }
        out_4d(b,m,h,w) = sum;
    }             

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                   const float *host_mask, float **device_output_ptr,
                                                   float **device_input_ptr, float **device_mask_ptr,
                                                   int Batch, int Map_out, int Channel, int Height, int Width, int K) {
    printf("Task: final_op\n");
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    std::cout << "mask size (bytes): " << mask_size << std::endl;
    std::cout << "K: " << K << std::endl;
    cudaMalloc(reinterpret_cast<void**>(device_input_ptr), input_size);
    cudaMalloc(reinterpret_cast<void**>(device_output_ptr), output_size);
    // cudaMalloc(reinterpret_cast<void**>(device_mask_ptr), mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    cudaError_t err = cudaMemcpyToSymbol(const_mask, host_mask, mask_size, 0,cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol error: " << cudaGetErrorString(err) << std::endl;
    }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                           const float *device_mask, int Batch, int Map_out,
                                           int Channel, int Height, int Width, int K) {
    // Set the kernel dimensions and call the kernel
    int Width_out = Width-K+1;
    int Height_out = Height-K+1;

    int W_grid = ceil(1.0*Width_out/tile_width);
    int H_grid = ceil(1.0*Height_out/tile_width);
    int image_size = H_grid * W_grid;
    int block_size = tile_width*tile_width;

    int batch_x = ceil(sqrt(Batch));
    int batch_y = ceil(Batch / batch_x);

    dim3 DimBlock(tile_width, tile_width, 1);

    // dim3 DimGrid(Map_out,W_grid*H_grid,Batch);
    dim3 DimGrid(H_grid * W_grid * Batch, Map_out, 1); 
    printf("%d, %d, %d\n",H_grid, W_grid, Batch);
    matmul_conv_fused_final<<<DimGrid, DimBlock>>>(device_input, device_output, Batch, Map_out, Channel, Height, Width, K, H_grid, W_grid, image_size);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    int Batch, int Map_out, int Channel, int Height, int Width, int K) {
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}

__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}