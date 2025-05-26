// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)
//@@ insert code here
__global__ void float_to_uchar(float *inputImage, unsigned char *ucharImage, int size){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<size) ucharImage[idx] = (unsigned char) (255 * inputImage[idx]);
}

__global__ void rgb_to_gray(unsigned char *ucharImage, unsigned char *grayImage, int height, int width, int size){
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  unsigned char r,g,b;
  if(col<width && row<height){
    int idx = row*width+col;
    if(idx<size){
      // here channels is 3
      r = ucharImage[3*idx];
      g = ucharImage[3*idx + 1];
      b = ucharImage[3*idx + 2];
      grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
    }
  }

}

__global__ void compute_histogram(unsigned char *grayImage, unsigned int *histogram, int size){
  __shared__ unsigned int temp[HISTOGRAM_LENGTH];
  int idx = blockIdx.x*blockDim.x+threadIdx.x;

  // init shared memory
  if(threadIdx.x<HISTOGRAM_LENGTH) temp[threadIdx.x] = 0;
  __syncthreads();

  if(idx<size){
    atomicAdd(&temp[grayImage[idx]], 1);
  }
  __syncthreads();

  //sum all histogram value and store it back to device memory
  if(threadIdx.x<HISTOGRAM_LENGTH){
    atomicAdd(&histogram[threadIdx.x], temp[threadIdx.x]);
  }
}
// compute cdf
__global__ void scan(unsigned int *input, float *cdf, int len, int size) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float prefixsum[BLOCK_SIZE*2];
  int id = threadIdx.x;
  int global_id = blockIdx.x * blockDim.x + id;
  
  if (global_id < len) {
    prefixsum[id] = input[global_id];
  } else {
    prefixsum[id] = 0;
  }
  __syncthreads();

  // Upsweep (Reduction)
  for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
    int index = (id + 1) * stride * 2 - 1;
    if (index < BLOCK_SIZE*2 && index-stride>=0) {
      prefixsum[index] += prefixsum[index - stride];
    }
    __syncthreads();
  }

  // Downsweep Phase
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
    int index = (id + 1) * stride * 2 - 1;
    if (index+stride < 2*BLOCK_SIZE) {
      prefixsum[index + stride] += prefixsum[index];
    }
    __syncthreads();
  }
  if (global_id < len) {
    cdf[global_id] = (prefixsum[id]*1.0/size);
  }
}
// apply correction
__global__ void uchar_correction(unsigned char *ucharImage, float *cdf, int size){
  float cdfMin = cdf[0];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float corrected = 255.0f * (cdf[ucharImage[idx]] - cdfMin) / (1.0f - cdfMin);
    // Clamp values
    corrected = fminf(fmaxf(corrected, 0.0f), 255.0f);
    ucharImage[idx] = (unsigned char)corrected;
  }
}

__global__ void uchar_to_float(unsigned char *ucharImage, float *outputImage, int size){
  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<size) outputImage[idx] = (float) (ucharImage[idx]/255.0f);
}
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  // Device memory pointers
  float *deviceInputImageData;
  float *deviceOutputImageData;
  unsigned char *deviceUCharImage;
  unsigned char *deviceGrayImage;
  unsigned int *deviceHistogram;
  float *deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  
  //@@ insert code here
  int imageSize = imageHeight*imageWidth;
  int imageSizeWithChannels = imageSize*imageChannels;

  // Allocate device memory
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageSizeWithChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageSizeWithChannels * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceUCharImage, imageSizeWithChannels * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceGrayImage, imageSize * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float)));
  
  // Copy input image data to device
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageSizeWithChannels * sizeof(float), cudaMemcpyHostToDevice));
  // Initialize histogram array to zeros
  wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  wbCheck(cudaMemset(deviceCDF, 0, HISTOGRAM_LENGTH * sizeof(unsigned int)));
  
  
  // Convert float to uchar
  dim3 DimGrid1(ceil(imageSizeWithChannels*1.0/BLOCK_SIZE),1,1);
  dim3 DimBlock1(BLOCK_SIZE,1,1);

  float_to_uchar<<<DimGrid1,DimBlock1>>>(deviceInputImageData,deviceUCharImage,imageSizeWithChannels);
  wbCheck(cudaDeviceSynchronize());

  // Convert uchar(rgb) to gray
  dim3 DimGrid2(ceil(imageWidth*1.0/16),ceil(imageHeight*1.0/16),1);
  dim3 DimBlock2(16,16,1);

  rgb_to_gray<<<DimGrid2,DimBlock2>>>(deviceUCharImage, deviceGrayImage,imageHeight,imageWidth,imageSize);
  wbCheck(cudaDeviceSynchronize());

  // Compute Historgram
  dim3 DimGrid3(ceil(imageSize*1.0/BLOCK_SIZE),1,1);
  dim3 DimBlock3(BLOCK_SIZE,1,1);

  compute_histogram<<<DimGrid3,DimBlock3>>>(deviceGrayImage,deviceHistogram,imageSize);
  wbCheck(cudaDeviceSynchronize());

  // Compute CDF from Histogram
  dim3 DimBlock4(BLOCK_SIZE,1,1);
  dim3 DimGrid4(ceil(HISTOGRAM_LENGTH*1.0/BLOCK_SIZE),1,1);

  scan<<<DimGrid4,DimBlock4>>>(deviceHistogram, deviceCDF, HISTOGRAM_LENGTH, imageSize);
  wbCheck(cudaDeviceSynchronize());

  uchar_correction<<<DimGrid1, DimBlock1>>>(deviceUCharImage, deviceCDF, imageSizeWithChannels);
  wbCheck(cudaDeviceSynchronize());

  uchar_to_float<<<DimGrid1,DimBlock1>>>(deviceUCharImage,deviceOutputImageData, imageSizeWithChannels);
  wbCheck(cudaDeviceSynchronize());

  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSizeWithChannels * sizeof(float), cudaMemcpyDeviceToHost);

  // Set output image data
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  // Free device memory
  wbCheck(cudaFree(deviceInputImageData));
  wbCheck(cudaFree(deviceOutputImageData));
  wbCheck(cudaFree(deviceUCharImage));
  wbCheck(cudaFree(deviceGrayImage));
  wbCheck(cudaFree(deviceHistogram));
  wbCheck(cudaFree(deviceCDF));
  
  // Free host memory
  wbImage_delete(inputImage);
  wbImage_delete(outputImage);
  return 0;
}

