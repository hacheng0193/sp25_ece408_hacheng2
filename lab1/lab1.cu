// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id<len) out[id] = in1[id]+in2[id];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  wbLog(TRACE, "The input length is ", inputLength);

  //@@ Allocate GPU memory here
  float *in1, *in2, *out;
  cudaMalloc((void **) &in1, inputLength*sizeof(float));
  cudaMalloc((void **) &in2, inputLength*sizeof(float));
  cudaMalloc((void **) &out, inputLength*sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(in1, hostInput1, inputLength*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(in2, hostInput2, inputLength*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here

  dim3 DimGrid(ceil(inputLength/1024.0), 1, 1);
  dim3 DimBlock(1024, 1, 1);
  vecAdd<<<DimGrid, DimBlock>>>(in1, in2, out, inputLength);

  //@@ Launch the GPU Kernel here to perform CUDA computation

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, out, inputLength*sizeof(float), cudaMemcpyDeviceToHost);


  //@@ Free the GPU memory here
  cudaFree(in1);
  cudaFree(in2);
  cudaFree(out);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
