#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define mask_size 27
//@@ Define constant memory for device kernel here
__constant__ float mask[mask_size];


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  // 3d to 1d transformation
  // output[z][y][x] = output[z*stepz+y*stepy+x]
  int stepz = x_size*y_size;
  int stepy = x_size;
  float sum=0;

  if(x<x_size && y<y_size && z<z_size){
    for(int i=-1;i<=1;i++){
      for(int j=-1;j<=1;j++){
        for(int k=-1;k<=1;k++){
          if((z+i) >= 0 && (z+i) < z_size && (y+j) >= 0 && (y+j) < y_size && (x+k) >= 0 && (x+k) < x_size){
            sum += input[(z+i)*stepz+(y+j)*stepy+(x+k)] * mask[(i+1)*9+(j+1)*3+(k+1)];
          }
        }
      }
    }
    output[z*stepz+y*stepy+x] = sum;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, (inputLength-3)*sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength-3)*sizeof(float));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3)*sizeof(float), cudaMemcpyHostToDevice);
  wbCheck(cudaMemcpyToSymbol(mask, hostKernel, mask_size*sizeof(float), 0,cudaMemcpyHostToDevice));


  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(1.0*x_size/8), ceil(1.0*y_size/8), ceil(1.0*z_size/8));
  dim3 DimBlock(8,8,8);

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size,y_size,x_size);
  wbCheck(cudaDeviceSynchronize());


  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3)*sizeof(float), cudaMemcpyDeviceToHost));



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);


  // Free host memory
  free(hostInput);
  free(hostOutput);
  
  return 0;
}

