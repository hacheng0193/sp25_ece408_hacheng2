// MP5 Reduction
// Input: A num list of length n
// Output: Sum of the list = list[0] + list[1] + ... + list[n-1];

#include <wb.h>

#define BLOCK_SIZE 32 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  //@@ Load a segment of the input vector into shared memory
  //@@ Traverse the reduction tree
  //@@ Write the computed sum of the block to the output vector at the correct index
  __shared__ float partialsum[BLOCK_SIZE];
  int t = threadIdx.x;
  int s = blockIdx.x*blockDim.x*2+t;
  if(s<len){
    if(s+BLOCK_SIZE<len){
      partialsum[t] = input[s]+input[s+BLOCK_SIZE];
    }
    else partialsum[t] = input[s];
  }
  else{
    partialsum[t] = 0;
  }
  __syncthreads();
  /*
  for(unsigned int stride=blockDim.x/2; stride>0;stride>>=1){
    if(t<stride){
        partialsum[t] += partialsum[t+stride];
    }
    __syncthreads();
  }
  */
  for(unsigned int stride=1; stride<BLOCK_SIZE;stride*=2){
    if(t%stride==0){
        partialsum[2*t] += partialsum[2*t+stride];
    }
    __syncthreads();
  }
  if(t==0){
    output[blockIdx.x] = partialsum[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  //Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements

  //@@ Allocate GPU memory
  float *deviceinput, *deviceoutput;
  cudaMalloc((void **) &deviceinput, numInputElements*sizeof(float));
  cudaMalloc((void **) &deviceoutput, numOutputElements*sizeof(float));

  //@@ Copy input memory to the GPU
  cudaMemcpy(deviceinput, hostInput, numInputElements*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE,1,1);
  dim3 DimGrid(ceil(numInputElements*1.0/BLOCK_SIZE),1,1);

  //@@ Launch the GPU Kernel and perform CUDA computation
  total<<<DimGrid,DimBlock>>>(deviceinput, deviceoutput, numInputElements);
  
  cudaDeviceSynchronize();  
  //@@ Copy the GPU output memory back to the CPU
  cudaMemcpy(hostOutput, deviceoutput, numOutputElements*sizeof(float), cudaMemcpyDeviceToHost);
  
  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. 
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }
  //@@ Free the GPU memory
  cudaFree(deviceinput);
  cudaFree(deviceoutput);


  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

