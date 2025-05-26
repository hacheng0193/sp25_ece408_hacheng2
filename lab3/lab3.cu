#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int by = blockIdx.y, ty = threadIdx.y;
  int bx = blockIdx.x, tx = threadIdx.x;
  int r = by*TILE_WIDTH+ty;
  int c = bx*TILE_WIDTH+tx;

  float sum = 0;
  for(int i=0;i<ceil(1.0*numAColumns/TILE_WIDTH);i++){
    if(r < numCRows && i * TILE_WIDTH + tx < numAColumns) 
      subTileA[ty][tx] = A[r * numAColumns + i * TILE_WIDTH + tx];
    else 
      subTileA[ty][tx] = 0;

    if(c < numCColumns && i * TILE_WIDTH + ty < numBRows) 
      subTileB[ty][tx] = B[(i * TILE_WIDTH + ty) * numBColumns + c];
    else 
      subTileB[ty][tx] = 0;

    __syncthreads();
    for(int k=0;k<TILE_WIDTH;k++){
      sum+=subTileA[ty][k]*subTileB[k][tx];
    }
    __syncthreads();
  }
  if(r < numCRows && c < numCColumns){
    C[r * numCColumns + c] = sum;
  }
  

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  //wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows*numCColumns * sizeof(float));\

  //@@ Allocate GPU memory here
  float *a, *b, *c;
  cudaMalloc((void **) &a, numARows*numAColumns*sizeof(float));
  cudaMalloc((void **) &b, numBRows*numBColumns*sizeof(float));
  cudaMalloc((void **) &c, numCRows*numCColumns*sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(a, hostA, numARows*numAColumns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b, hostB, numBRows*numBColumns*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(1.0*numCColumns/TILE_WIDTH), ceil(1.0*numCRows/TILE_WIDTH), 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(a, b, c, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, c, numCRows*numCColumns*sizeof(float), cudaMemcpyDeviceToHost);


  //@@ Free the GPU memory here
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);


  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix

  return 0;
}
