#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"

#define PERMUTE_BLOCK_SIZE 256
#define tile_width 16

__global__ void matrix_unrolling_kernel_stream_with_batch(const float *input, float *output,const int Batch, const int Channel,const int Height, const int Width,const int K) { // the kernel is local to just one image, Batch=1
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    int b = 0; // only one image per batch
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    size_t Width_unrolled = Batch*Height_out*Width_out;
    size_t W_grid = ceil(1.0*Width_out/tile_width);

    size_t h = blockIdx.y * tile_width + threadIdx.y;
    size_t w = blockIdx.x * tile_width + threadIdx.x;
    
    size_t col_unroll = b * (Height_out * Width_out) + (h * Width_out + w);
    if(h<Height_out && w<Width_out){
        for (int c = 0; c < Channel; ++c) { // for each input channel
            size_t w_base = c*(K*K); // per-channel offset for smallest X_unroll index
            for (int p = 0; p < K; ++p){// for each element of KxK filter (two loops)
                for (int q = 0; q < K; ++q) {
                    // for each thread (each output value, two loops)
                    size_t row_unroll = w_base+p*K+q;

                    output[row_unroll*Width_unrolled+col_unroll] = in_4d(b,c,h+p,w+q);// copy input pixels
                }
            }
        }
    }
    
    #undef in_4d
}


// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel_stream_with_batch(const float *input, float *output, int Map_out, int Batch, int image_size) {
    int b=0;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);  

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Height_out * Width_out;

    printf("Task: req_0\n");
    // Allocate memory for full input/output/mask 
    cudaMalloc(device_input_ptr, input_size);
    cudaMalloc(device_output_ptr, output_size);
    cudaMalloc(device_mask_ptr, mask_size);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    
    const int NUM_STREAMS = 8; // Allocate per-batch buffers
    float *unrolled_per_batch[NUM_STREAMS] ;
    float *output_per_batch[NUM_STREAMS] ;

    
    cudaStream_t streams[NUM_STREAMS];
    // 8 streams
    for (int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&unrolled_per_batch[i], Height_unrolled * Width_unrolled * sizeof(float));
        cudaMalloc(&output_per_batch[i], Map_out * Width_unrolled * sizeof(float));
    }

    int input_offset = Channel * Height * Width;
    int output_offset = Map_out * Height_out * Width_out;

    dim3 DimBlock(tile_width, tile_width, 1);
    dim3 DimGrid(ceil((float)Width_out / tile_width), ceil((float)Height_out / tile_width), 1);
    dim3 matmul_grid_dim((Width_unrolled - 1) / MATMUL_TILE_WIDTH + 1,
                        (Map_out - 1) / MATMUL_TILE_WIDTH + 1, 1);
    dim3 matmul_block_dim(MATMUL_TILE_WIDTH, MATMUL_TILE_WIDTH, 1);
    dim3 permute_kernel_grid_dim((Width_unrolled - 1) / PERMUTE_BLOCK_SIZE + 1, Batch, 1);

    //  copy first batch data
    cudaMemcpyAsync(*device_input_ptr, host_input,
                Channel * Height * Width * sizeof(float),
                cudaMemcpyHostToDevice, streams[0]); // stream 0 for copy

    int streamid;
    for (int b = 0; b < Batch; ++b) {
        streamid = b % NUM_STREAMS;
        
        float *input_b = *device_input_ptr + b * input_offset;
        float *output_b = *device_output_ptr + b * output_offset;
        const float *host_input_b = host_input + b * input_offset;
        
        // Prefetch next batch 
        if (b < Batch - 1) {
            int next_streamid = (b + 1) % NUM_STREAMS;
            float *input_next_b = *device_input_ptr + (b + 1) * input_offset;
            const float *host_input_next_b = host_input + (b + 1) * input_offset;
            
            cudaMemcpyAsync(input_next_b, host_input_next_b,
                            Channel * Height * Width * sizeof(float),
                            cudaMemcpyHostToDevice, streams[next_streamid]);
        }

        // Compute kernels 
        matrix_unrolling_kernel_stream_with_batch<<<DimGrid, DimBlock, 0, streams[streamid]>>>(
            input_b, unrolled_per_batch[streamid], 1,
            Channel, Height, Width, K
        );

        matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim, 0, streams[streamid]>>>(
            *device_mask_ptr, unrolled_per_batch[streamid], output_per_batch[streamid],
            Map_out, Height_unrolled,
            Height_unrolled, Width_unrolled,
            Map_out, Width_unrolled
        );

        matrix_permute_kernel_stream_with_batch<<<permute_kernel_grid_dim, PERMUTE_BLOCK_SIZE, 0, streams[streamid]>>>(
            output_per_batch[streamid], output_b,
            Map_out, 1, Width_unrolled
        );
    }

    cudaDeviceSynchronize();

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
        cudaFree(unrolled_per_batch[i]);
        cudaFree(output_per_batch[i]);
    }
    // Error check
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // all done in conv_forward_gpu_prolog
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    size_t input_size = Batch*Channel*Height*Width*sizeof(float);
    size_t output_size = Batch*Map_out*(Height - K + 1)*(Width - K + 1)*sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    
    // TODO: Free device memory
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