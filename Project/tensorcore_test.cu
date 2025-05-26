#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <random>
using namespace nvcuda;

#define TILE_WIDTH 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

__global__ void matmul_tf32_tensor_core_1(float *A, float *B, float *C, int M, int K, int N) {
    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    extern __shared__ float shared[];
    float* tile_A = shared;                                
    float* tile_B = tile_A + 2*WMMA_M * WMMA_K;                        
    float* tile_C = tile_B + 2*WMMA_K * WMMA_N;                        

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // 重要：把 accumulator fragment 放到 register
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int tileId = 0; tileId < (K + WMMA_K*2 - 1) / (WMMA_K*2); tileId++) {
        int tiledIdx = tileId * WMMA_K * 2;

        // 讀 A
        if (row < M && (tiledIdx + tx) < K) {
            int group = tx / 8;
            tile_A[group * WMMA_M * WMMA_K + ty * WMMA_K + (tx % 8)] = A[row * K + (tiledIdx + tx)];
        } else {
            int group = tx / 8;
            tile_A[group * WMMA_M * WMMA_K + ty * WMMA_K + (tx % 8)] = 0.0f;
        }

        // 讀 B
        if (col < N && (tiledIdx + ty) < K) {
            int group = ty / 8;
            tile_B[group * WMMA_K * WMMA_N + (ty % 8) * WMMA_N + tx] = B[(tiledIdx + ty) * N + col];
        } else {
            int group = ty / 8;
            tile_B[group * WMMA_K * WMMA_N + (ty % 8) * WMMA_N + tx] = 0.0f;
        }

        __syncthreads();

        // 每個 warp 做計算
        if (ty < 2) {
            for (int t = 0; t < 2; t++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, tile_A + t * WMMA_M * WMMA_K, WMMA_K);
                wmma::load_matrix_sync(b_frag, tile_B + t * WMMA_K * WMMA_N, WMMA_N);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
        }
        __syncthreads();
    }

    // 寫回 C
    if (ty < 2) {
        wmma::store_matrix_sync(tile_C, c_frag, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    if (row < M && col < N) {
        C[row * N + col] = tile_C[ty * TILE_WIDTH + tx];
    }
}

__global__ void matmul_tf32_tensor_core(float *A, float *B, float *C, int M, int K, int N) {
    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    extern __shared__ float shared[];
    float* tile_A = shared;                                
    float* tile_B = tile_A + 2*WMMA_M * WMMA_K;                        
    float* tile_C = tile_B + 2*WMMA_K * WMMA_N;                        

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag_1, a_frag_2;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag_1,b_frag_2;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag_1,c_frag_2;
    wmma::fill_fragment(c_frag_1, 0.0f);
    wmma::fill_fragment(c_frag_2, 0.0f);

    for (int tileId = 0; tileId < (K + TILE_WIDTH - 1) / TILE_WIDTH; tileId++) {
        int tiledIdx = tileId * TILE_WIDTH;
        
        // Load A
        if (row < M && (tiledIdx + tx) < K) {
            int group = tx / 8;
            tile_A[group * WMMA_M * WMMA_K + ty * WMMA_K + (tx % 8)] = A[row * K + (tiledIdx + tx)];
        } else {
            int group = tx / 8;
            tile_A[group * WMMA_M * WMMA_K + ty * WMMA_K + (tx % 8)] = 0.0f;
        }

        // Load B
        if (col < N && (tiledIdx + ty) < K) {
            int group = ty / 8;
            tile_B[group * WMMA_K * WMMA_N + (ty % 8) * WMMA_N + tx] = B[(tiledIdx + ty) * N + col];
        } else {
            int group = ty / 8;
            tile_B[group * WMMA_K * WMMA_N + (ty % 8) * WMMA_N + tx] = 0.0f;
        }

        __syncthreads();

        if (ty==0) {
            wmma::load_matrix_sync(a_frag_1, tile_A, WMMA_K);
            wmma::load_matrix_sync(b_frag_1, tile_B, WMMA_N);
            wmma::mma_sync(c_frag_1, a_frag_1, b_frag_1, c_frag_1);
        }
        else if (ty==1) {
            wmma::load_matrix_sync(a_frag_2, tile_A+WMMA_M*WMMA_K, WMMA_K);
            wmma::load_matrix_sync(b_frag_2, tile_B+WMMA_K*WMMA_N, WMMA_N);
            wmma::mma_sync(c_frag_2, a_frag_2, b_frag_2, c_frag_2);
        }
        __syncthreads();
    }

    if (ty == 0) {
        wmma::store_matrix_sync(tile_C, c_frag_1, WMMA_N, wmma::mem_row_major);
    }
    else if (ty == 1) {
        wmma::store_matrix_sync(tile_C + WMMA_M * WMMA_K, c_frag_2, WMMA_N, wmma::mem_row_major);
    }

    __syncthreads();

    if (row < M && col < N) {
        C[row * N + col] = tile_C[ty * TILE_WIDTH + tx]+tile_C[WMMA_M * WMMA_K+ty * TILE_WIDTH + tx];
    }
}

void matmul_cpu(const float *A, const float *B, float *C, int M, int K, int N) {
    // CPU version of matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

int main() {
    // Using dimensions that match your requirements
    int M = 30, K = 30, N = 30;
    float *A, *B, *C, *C_cpu;
    float *d_A, *d_B, *d_C;

    // Allocate memory
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));
    C_cpu = (float *)malloc(M * N * sizeof(float));

    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    
    // Fill matrix A with random integers
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            // A[i * K + j] = M*K-i * K + j;
            A[i * K + j] = 1;
        }
    }

    // Fill matrix B with random integers
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            // B[i * N + j] = i * N + j;
            B[i * N + j] = 1;
        }
    }

    // Initialize C matrices to zero
    memset(C, 0, M * N * sizeof(float));
    memset(C_cpu, 0, M * N * sizeof(float));


    // Copy A and B to GPU
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Each warp computes a 16x16 output tile
    dim3 gridDim((N + WMMA_N - 1) / 16, (M + WMMA_M - 1) / 16);
    dim3 blockDim(16, 16);  // One warp per block for simplicity
    size_t shmem_size = (2*WMMA_M * WMMA_K + 2*WMMA_K * WMMA_N + 2*WMMA_M * WMMA_N) * sizeof(float);
    // Launch kernel
    matmul_tf32_tensor_core<<<gridDim, blockDim,shmem_size>>>(d_A, d_B, d_C, M, K, N);

    // Check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU version for validation
    matmul_cpu(A, B, C_cpu, M, K, N);

    // Compare GPU and CPU results
    bool match = true;
    float max_diff = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float diff = fabs(C[i * N + j] - C_cpu[i * N + j]);
            max_diff = std::max(diff, max_diff);
            if (diff > 1e-1) {  // Larger tolerance for TF32 precision
                match = false;
                std::cout << "Mismatch at [" << i << "][" << j << "]: GPU=" 
                      << C[i * N + j] << ", CPU=" << C_cpu[i * N + j] << std::endl;
                break;
            }
        }
        if (!match) break;
    }

    if (match) {
        std::cout << "Test passed: GPU and CPU results match (max diff: " << max_diff << ")!" << std::endl;
    } else {
        std::cout << "Test failed: GPU and CPU results do not match (max diff: " << max_diff << ")!" << std::endl;
    }

    // Print a few values from the result matrices for verification
    std::cout << "\nSample values from GPU result matrix C:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSample values from CPU result matrix C_cpu:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C_cpu[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up memory
    free(A);
    free(B);
    free(C);
    free(C_cpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}