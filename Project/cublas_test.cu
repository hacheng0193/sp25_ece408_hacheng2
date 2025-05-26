#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS(call) \
    if ((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __LINE__ << std::endl; exit(EXIT_FAILURE); \
    }

int main() {
    const int M = 2, K = 3, N = 3;  // A: MxK, B: KxN => C: MxN (row-major)
    
    float h_A[M*K] = {1, 2, 3,   // A = [ [1,2,3],
                       4, 5, 6};  //       [4,5,6] ]
    
    float h_B[K*N] = {1, 1, 1,     // B = [ [1,4],
                      1, 1, 1,     //       [2,5],
                      1, 1, 1};     //       [3,6] ]
    
    float h_C[M*N] = {0};         // Output matrix

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&d_B, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&d_C, sizeof(float) * M * N));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    // A and B are row-major, simulate by transposing both
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,           // Result is MxN, so rows=N, cols=M in column-major
        &alpha,
        d_B, N,            // B is KxN in row-major, becomes NxK column-major
        d_A, K,            // A is MxK in row-major, becomes KxM column-major
        &beta,
        d_C, N             // C is MxN in row-major, store as N x M column-major
    ));

    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    std::cout << "Result C = A * B:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
