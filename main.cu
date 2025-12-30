#include <stdio.h>
#include <assert.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <sstream>
#include <fstream>
#include <vector>

#include <string.h>
#include <stdlib.h> // For malloc and free

#include "matrixmul.h"

#define IDX2C(i,j,ld) (((i)*(ld))+(j))


using namespace std;

inline
cudaError_t checkCudaErrors(cudaError_t result, string functioncall = "")
{
//#if defined(DEBUG) || defined(_DEBUG)
  //fprintf(stderr, "CUDA Runtime Error: %d\n", result);
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error for this function call ( %s ) : %s\n", 
            functioncall.c_str(), cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
//#endif
  return result;
}


void readfile (const char* filename, float* data) {
  
  
  FILE *file;
  char line[256]; // Buffer to store each line
  char *token;

  file = fopen(filename, "r"); 
  if (file == NULL) {
     fprintf(stderr, "Error opening file");
     return;
  }
  int count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      // Remove trailing newline character if present
      line[strcspn(line, "\n")] = 0; 
      // Split the line by spaces
      token = strtok(line, " ");
      while (token != NULL) {
          float t = atof(token);
          data[count] = t;
	  token = strtok(NULL, " ");
	  count+=1;        
      }
    }
  fclose(file); 
}	


int
main( int argc, char* argv[ ] )
{ 
  //srand(time(0));
//  fprintf (stderr, "Amount of data transfered to the device is %lld GB\n", bytes4euc/1000000000);
  
  vector<char*> array;
  

  int numberActivation;
  int numberActivation2;
  int numberActivation3;
  int numberOfFeatures;
  int numData;   


  if (argc > 5) {	  
      numData = atoi(argv[1]);   
      numberOfFeatures = atoi(argv[2]);   
      numberActivation = atoi(argv[3]);  
      numberActivation2 = atoi(argv[4]);  
      numberActivation3 = atoi(argv[5]);  
  } else {
      numberActivation = 5;
      numberActivation2 = 3;
      numberActivation3 = 1;
      numberOfFeatures = 10;
      numData = 2;   
  } 
  int wRows = numberActivation;
  int xCols = numData;
  int wColsXRows = numberOfFeatures;
  
  
  int BLOCKSIZE = 758;
  int NUMBLOCKS = ((wRows * numData)+BLOCKSIZE-1)/BLOCKSIZE;
    
  
  
  // checking memory coalescing
   
  float* xxData = new float[numData * numberOfFeatures];
  
  for (int i = 0; i < (numData * numberOfFeatures); i++) {
 //     xxData[i] = (rand() % 8)/1.0; 
  }
  
 
  float* yData = new float[numData];
  
  float* weightBiasArray = new float[numberActivation * (numberOfFeatures+1)];
  float* weightBiasArray2 = new float[(numberActivation + 1) * numberActivation2];
  float* weightBiasArray3 = new float[(numberActivation2 + 1) * numberActivation3];
  
  
  // Reading the Pytorch initial weight and biases for the first layer 
  readfile ("weightbias1.txt", weightBiasArray);
  
  
  // Reading the Pytorch initial weight and biases for the second layer 
  readfile ("weightbias2.txt", weightBiasArray2);
    
  // Reading the Pytorch initial weight and biases for the third layer 
  readfile ("weightbias3.txt", weightBiasArray3);
  
  // Reading the input data 
   readfile ("inputdata.txt", xxData);
 
  // Reading the target data 
  readfile ("targetdata.txt", yData);

  // Allocate memory on device
  float *xxDataDev;
  float *weightBiasArrayDev;
  float *actValuesDev;
  // Allocate memory on host
  float *HactValues = new float[wRows * numData];
   
   
  // Create CUDA events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // forward propagation 

  //  doing first layer linear1  

  cudaError_t status; 
  int weightBiasArraySize = (sizeof(float) * numberActivation * (numberOfFeatures+1)) ;
  //fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", weightBiasArraySize);
   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&weightBiasArrayDev), weightBiasArraySize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&weightBiasArrayDev), weightBiasArraySize)"); 

  //allocate memory on the GPU device
  int xxDataSize = (sizeof(float) * (numData * numberOfFeatures)); 
  //fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", xxDataSize);
  status = cudaMalloc( (void **)(&xxDataDev), xxDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&xxDataDev), xxDataSize)");
  
  //allocate memory on the GPU device
  int actValuesSize = (sizeof(float) * (numData * wRows)); 
  //fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", actValuesSize);
  status = cudaMalloc( (void **)(&actValuesDev), actValuesSize);
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMalloc( (void **)(&actValuesDev), actValuesSize) ");
  

  // copy data from host memory to the device:

  status = cudaMemcpy(weightBiasArrayDev, weightBiasArray, weightBiasArraySize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(weightBiasArrayDev, weightBiasArray, weightBiasArraySize, cudaMemcpyHostToDevice )" );  


 // copy data from host memory to the device:

  status = cudaMemcpy(xxDataDev, xxData, xxDataSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," cudaMemcpy(xxDataDev, xxData, xxDataSize, cudaMemcpyHostToDevice )" );
  
   
  // allocate number of threads in a block  
  dim3 threads(BLOCKSIZE, 1, 1 );

  // allocate number of blocks
  dim3 grid(NUMBLOCKS, 1, 1 );
  

  // Record the start event
  cudaEventRecord(start, 0); 

  // call the kernel
  matrixMulAddRowBasedARR2<<< grid, threads >>>( weightBiasArrayDev, xxDataDev, actValuesDev, wRows, xCols, wColsXRows);
  
  status = cudaDeviceSynchronize( );
  
  // Record the stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop); 
  
  // Calculate elapsed time
  float GpuTime;
  cudaEventElapsedTime(&GpuTime, start, stop); 
  //printf("  GPU time: %f milliseconds\n", GpuTime); 
  
  checkCudaErrors( status,"matrixMulAddColBasedARR<<< grid, threads >>>( weightBiasArrayDev, xxDataDev, actValuesDev, wRows, xCols, wColsXRows) ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  // copy data device memory to host:
  cudaMemcpy(HactValues, actValuesDev,  actValuesSize , cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );
  
  
  // doing the relu calculation 

  float *HactValuesRelu = new float[wRows * numData];

  
  BLOCKSIZE = 758;
  NUMBLOCKS = ((wRows * numData)+BLOCKSIZE-1)/BLOCKSIZE;
 
  threads.x = BLOCKSIZE;
  grid.x = NUMBLOCKS; 

  matrixReLu<<< grid, threads >>>(actValuesDev, (wRows * numData));   
  
  checkCudaErrors( status,"matrixReLu<<< grid, threads >>>(actValuesDev, (wRows * xCols))");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  // copy data device memory to host:
  cudaMemcpy(HactValuesRelu, actValuesDev,  actValuesSize , cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );
  
 // doing second linear layer 
   
  float *HactValues2 = new float[numberActivation2  * numData];

  
  BLOCKSIZE = 758;
  NUMBLOCKS = ((numberActivation2  * numData)+BLOCKSIZE-1)/BLOCKSIZE;
 
  threads.x = BLOCKSIZE;
  grid.x = NUMBLOCKS; 

  
  
  float *weightBiasArrayDev2;
  float *actValuesDev2;

  int weightBiasArraySize2 = (sizeof(float) * numberActivation2 * (numberActivation+1)) ;
  //fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", weightBiasArraySize2);
   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&weightBiasArrayDev2), weightBiasArraySize2);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&weightBiasArrayDev2), weightBiasArraySize2)"); 

  //allocate memory on the GPU device
  int actValuesDevSize2 = (sizeof(float) * (numData * numberActivation2)); 
 // fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", actValuesDevSize2);
  status = cudaMalloc( (void **)(&actValuesDev2), actValuesDevSize2);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&actValuesDevSize2), actValuesDevSize2) ");
  
  
  
  // copy data from host memory to the device:

  status = cudaMemcpy(weightBiasArrayDev2, weightBiasArray2, weightBiasArraySize2, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(weightBiasArrayDev2, weightBiasArray2, weightBiasArraySize2, cudaMemcpyHostToDevice )" );  

    
  
  // call the kernel
  matrixMulAddRowBasedARR2<<< grid, threads >>>( weightBiasArrayDev2, actValuesDev, actValuesDev2, numberActivation2, numData, numberActivation);
  
  status = cudaDeviceSynchronize( );
  
  checkCudaErrors( status,"matrixReLu<<< grid, threads >>>(actValuesDev, (wRows * xCols))");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  // copy data device memory to host:
  cudaMemcpy(HactValues2, actValuesDev2,  actValuesDevSize2, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );
 
  
  // doing the second relu calculation 

  float *HactValuesRelu2 = new float[numberActivation2 * numData];

  
  BLOCKSIZE = 758;
  NUMBLOCKS = ((numberActivation2 * numData)+BLOCKSIZE-1)/BLOCKSIZE;
 
  threads.x = BLOCKSIZE;
  grid.x = NUMBLOCKS; 

  matrixReLu<<< grid, threads >>>(actValuesDev2, (numberActivation2 * numData));   
  
  checkCudaErrors( status,"matrixReLu<<< grid, threads >>>(actValuesDev2, (numberActivation2 * xCols))");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  // copy data device memory to host:
  cudaMemcpy(HactValuesRelu2, actValuesDev2,  actValuesDevSize2, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );

 // doing third linear layer 
  
  float *HactValues3 = new float[numberActivation3  * numData];

  
  BLOCKSIZE = 758;
  NUMBLOCKS = ((numberActivation3  * numData)+BLOCKSIZE-1)/BLOCKSIZE;
 
  threads.x = BLOCKSIZE;
  grid.x = NUMBLOCKS; 
  
     
  float *weightBiasArrayDev3;
  float *actValuesDev3;

  int weightBiasArraySize3 = (sizeof(float) * numberActivation3 * (numberActivation2+1)) ;
  //fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", weightBiasArraySize2);
   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&weightBiasArrayDev3), weightBiasArraySize3);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&weightBiasArrayDev2), weightBiasArraySize2)"); 

  //allocate memory on the GPU device
  int actValuesDevSize3 = (sizeof(float) * (numData * numberActivation3)); 
 // fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", actValuesDevSize2);
  status = cudaMalloc( (void **)(&actValuesDev3), actValuesDevSize3);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&actValuesDevSize3), actValuesDevSize3) ");
  
  
  
  // copy data from host memory to the device:

  status = cudaMemcpy(weightBiasArrayDev3, weightBiasArray3, weightBiasArraySize3, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"cudaMemcpy(weightBiasArrayDev3, weightBiasArray3, weightBiasArraySize3, cudaMemcpyHostToDevice )" );  

    
  
  // call the kernel
  matrixMulAddRowBasedARR2<<< grid, threads >>>( weightBiasArrayDev3, actValuesDev2, actValuesDev3, numberActivation3, numData, numberActivation2);
  
  status = cudaDeviceSynchronize( );
  
  checkCudaErrors( status," matrixMulAddRowBasedARR2<<< grid, threads >>>( weightBiasArrayDev3, actValuesDev2, actValuesDev3, numberActivation3, numData, numberActivation2);");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  // copy data device memory to host:
  cudaMemcpy(HactValues3, actValuesDev3,  actValuesDevSize3, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status ); 
  
  
  // softmax 

  
  BLOCKSIZE = 758;
  NUMBLOCKS = ((numberActivation3  * numData)+BLOCKSIZE-1)/BLOCKSIZE;
 
  threads.x = BLOCKSIZE;
  grid.x = NUMBLOCKS; 
  
    
  matrixSigmoid<<< grid, threads >>>(actValuesDev3, (numberActivation3 * numData));   
   
  status = cudaDeviceSynchronize( );
    
  checkCudaErrors( status,"matrixBinarySoftmax<<< grid, threads >>>(actValuesDev2, (numberActivation2 * numData));");   
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  
  // copy data device memory to host:
  cudaMemcpy(HactValues3, actValuesDev3,  actValuesDevSize3, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );

  
  // reading pytorch output data
  // Reading the target data 
  float *HactValues3Pytorch = new float[numberActivation3  * numData]; 
  readfile("outputdata.txt", HactValues3Pytorch);
  fprintf(stderr, "Checking sigmoid output (prediction) match between Pytorch and CUDA implementation: "); 
  bool outputMatch = true;
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation3; j++) { 
	float torchval = HactValues3Pytorch[IDX2C(i,j,numberActivation3)];
        float cudaval = HactValues3[IDX2C(i,j,numberActivation3)];
        if ( (powf(torchval  - cudaval , 2.0)) > powf(0.01, 2.0) )  {
	   outputMatch = false;
	   break;
	}
      }
  } 
  if (outputMatch) {
   fprintf(stderr, "Yes, they match\n"); 
  } else {
   fprintf(stderr, "No, they do not match\n"); 
  }
  fprintf(stderr, " \n");
  // Backpropagation

  // dL/dZ3 = a-y, where a is the prediction and while y is the target

  
  //elementWiseSub<<< grid, threads >>>(first, (numberActivation3 * numData));   
  
  
  float* yDataDev;
  float* dL_dZ3 = new float [numberActivation3 * numData];
  float* dL_dZ3_Dev;
  
  
  float* dL_dZ3_Dev_T;

  float* dL_dZ3_T = new float [numData * numberActivation3]; 

  BLOCKSIZE = 758;
  NUMBLOCKS = ((numData)+BLOCKSIZE-1)/BLOCKSIZE;
  

  int yDataSize = (sizeof(float) * numberActivation3 * numData) ;
  
  int dL_dZ3_DevSize = yDataSize; 
  //fprintf (stderr, "Amount of data transfered to the device is %d Bytes\n", weightBiasArraySize2);
   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&yDataDev), yDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&yDataDev), yDataSize);"); 
  
  
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&dL_dZ3_Dev), yDataSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&actValuesRelu2Dev), activationDevSize2);"); 


  // copy data from host memory to the device:

  status = cudaMemcpy(yDataDev, yData, yDataSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," cudaMemcpy(yDataDev, yData, yDataSize, cudaMemcpyHostToDevice ); " );  
 
  // call the Kernel 
  elementWiseSub<<< grid, threads >>>(actValuesDev3, yDataDev, numData);    
 
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status," elementWiseSub<<< grid, threads >>>(actValuesDev3, yDataDev, numData); ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  

  status = cudaMemcpy(dL_dZ3_Dev, actValuesDev3, yDataSize, cudaMemcpyDeviceToDevice);
  checkCudaErrors( status, "  cudaMemcpy(dL_dZ3_Dev, actValuesDev3, yDataSize, cudaMemcpyDeviceToDevice);" );

  // copy data device memory to host:
  cudaMemcpy(dL_dZ3, actValuesDev3,  yDataSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(dL_dZ3, actValuesDev3,  yDataSize, cudaMemcpyDeviceToHost);  " );
  
  
  
  // Transpose of dL_dZ2 so it can be arranged in column major storage for matrix multiplication

  
  
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&dL_dZ3_Dev_T), dL_dZ3_DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&dL_dZ3_Dev_T), dL_dZ3_DevSize);"); 
  
  
  NUMBLOCKS = ((numData * (numberActivation3))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  matrixTransposeSubBias<<< grid, threads >>>(dL_dZ3_Dev, dL_dZ3_Dev_T, numberActivation3, numData); 

  // copy data device memory to host:
  cudaMemcpy(dL_dZ3_T, dL_dZ3_Dev_T, dL_dZ3_DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );
  
  
  // dL/dW3 USING ADAM optimizer at a learning rate of 0.001

  float* dL_dW3 = new float [numberActivation3 * (numberActivation2+1)];
  
  
  float* dL_dW3_Dev;
  
  float* actValuesRelu2Dev;
  float* actValuesRelu2Dev_T;
  float* diff_actValuesRelu2Dev;
  float* diff_HactValuesRelu2;
  float* diff_actValuesRelu2Dev_T;
  
  float* diff_HactValuesRelu2_T;

  float* HactValuesRelu2_T = new float [numData * (numberActivation2+1)];

  int actValuesDevSize2_T = sizeof(float) * numData * (numberActivation2+1); 
   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&actValuesRelu2Dev_T), actValuesDevSize2_T);
  // checks for cuda errors  
  checkCudaErrors( status, "cudaMalloc( (void **)(&yDataDev), yDataSize);"); 

   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&actValuesRelu2Dev), actValuesDevSize2);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&actValuesRelu2Dev), activationDevSize2);"); 
  
    


  // copy data from host memory to the device:
   
  status = cudaMemcpy(actValuesRelu2Dev, HactValuesRelu2, actValuesDevSize2, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," cudaMemcpy(actValuesRelu2Dev_T, HactValuesRelu2, activationDevSize2, cudaMemcpyHostToDevice )" );  
  
  
  NUMBLOCKS = ((numData * (numberActivation2+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  
  // the column of the bias was filled with one
  matrixTransposeAddBias<<< grid, threads >>>(actValuesRelu2Dev, actValuesRelu2Dev_T, numData, numberActivation2);    
  
  
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status,"  matrixTransposeAddBias<<< grid, threads >>>(actValuesRelu2Dev, actValuesRelu2Dev_T, numData, numberActivation2)");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  
  // copy data device memory to host:
  cudaMemcpy(HactValuesRelu2_T, actValuesRelu2Dev_T,  actValuesDevSize2_T, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(HactValuesRelu_T, actValuesRelu2Dev_T,  yDataSize, cudaMemcpyDeviceToHost); " );

   
  // differentiation of ReLu 2
  
  
  // this is to store the value of relu2 to be used for differentiation
  int diff_HactValuesRelu2DevSize = sizeof(float) * numData * (numberActivation2);  
  
  diff_HactValuesRelu2 = new float [numData * numberActivation2];
  
  diff_HactValuesRelu2_T = new float [numData * numberActivation2];

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&diff_actValuesRelu2Dev),  diff_HactValuesRelu2DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&diff_actValuesRelu2Dev),  diff_HactValuesRelu2DevSize); "); 

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&diff_actValuesRelu2Dev_T),  diff_HactValuesRelu2DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&diff_actValuesRelu2Dev),  diff_HactValuesRelu2DevSize); "); 


  
  status = cudaMemcpy(diff_actValuesRelu2Dev, actValuesRelu2Dev , diff_HactValuesRelu2DevSize, cudaMemcpyDeviceToDevice);
   
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMemcpy(diff_actValuesRelu2Dev, actValuesRelu2Dev , diff_HactValuesRelu2DevSize, cudaMemcpyDeviceToDevice); "); 
   // end of storing of the relu2 value
  
  NUMBLOCKS = ((numData * (numberActivation2))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  matrixDiffReLu<<< grid, threads >>>(diff_actValuesRelu2Dev, (numData * numberActivation2)); 
  
  // copy data device memory to host:
  cudaMemcpy(diff_HactValuesRelu2, diff_actValuesRelu2Dev,  diff_HactValuesRelu2DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, "cudaMemcpy(diff_HactValuesRelu2Dev, diff_actValuesRelu2Dev,  diff_HactValuesRelu2DevSize, cudaMemcpyDeviceToHost);");  
  
   
  matrixTranspose<<<grid, threads >>>(diff_actValuesRelu2Dev, diff_actValuesRelu2Dev_T, numData, numberActivation2); 

  // copy data device memory to host:
  cudaMemcpy(diff_HactValuesRelu2_T, diff_actValuesRelu2Dev_T,  diff_HactValuesRelu2DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, "cudaMemcpy(diff_HactValuesRelu2Dev_T, diff_actValuesRelu2Dev_T,  diff_HactValuesRelu2DevSize, cudaMemcpyDeviceToHost);");  
  
  
  // copy data from host memory to the device:
   
  status = cudaMemcpy(actValuesRelu2Dev, HactValuesRelu2, actValuesDevSize2, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," cudaMemcpy(actValuesRelu2Dev_T, HactValuesRelu2, activationDevSize2, cudaMemcpyHostToDevice )" );  
   
  
  int dL_dW3_size =  (sizeof(float) * (numberActivation2+1) * numberActivation3); 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&dL_dW3_Dev), dL_dW3_size);
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMalloc( (void **)(&dL_dW3_Dev), sizeof(float) * numberActivation2); "); 

    
  NUMBLOCKS = ((numberActivation3 * (numberActivation2+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;
  
  matrixdL_dW3<<< grid, threads >>>(actValuesDev3, actValuesRelu2Dev_T, dL_dW3_Dev, numberActivation3, (numberActivation2+1), numData);
  
  
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status,"  matrixdL_dW3<<< grid, threads >>>(actValuesDev3, actValuesRelu2Dev_T, dL_dW3_Dev, numberActivation3, (numberActivation2+1), numData)");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  // copy data device memory to host:
  cudaMemcpy(dL_dW3, dL_dW3_Dev,  dL_dW3_size, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(dL_dW3, dL_dW3_Dev,  dL_dW3_size, cudaMemcpyDeviceToHost);" ); 

    
  // Reading the dL_dW3 data
  float* dL_dW3_Pytorch = new float [numberActivation3 * (numberActivation2+1)];
  // reading pytorch dL_dW3 data
  readfile("weightbias3_grad.txt", dL_dW3_Pytorch);
  
  bool dL_dW3_Match = true;
  fprintf(stderr, "Checking dL_dW3 match between Pytorch and CUDA implementation: "); 
  for (int i = 0; i < numberActivation3; i++) {
   for (int j = 0; j <= numberActivation2; j++) {
        float torchval = dL_dW3_Pytorch[IDX2C(i,j,(numberActivation2 + 1))];
        float cudaval = dL_dW3[IDX2C(i,j,(numberActivation2 + 1))];
        if ( (powf(torchval  - cudaval , 2.0)) > powf(0.01, 2.0) )  {
           dL_dW3_Match =  false;
	   break;
	}
   }
  }
  if (dL_dW3_Match) {
     fprintf(stderr, "Yes, they match\n"); 
  } else {
     fprintf(stderr, "No, they do not match\n"); 
  }
   // transpose weightBiasArray3
    
  float* weightBiasArray3_T = new float [numberActivation3 * (numberActivation2+1)];
  
  float* weightBiasArrayDev3_T;

  int weightBiasArraySize3_T = (sizeof(float) * numberActivation3 * (numberActivation2+1));  
  
  // allocate memory on the GPU;
  status = cudaMalloc( (void **)(&weightBiasArrayDev3_T), weightBiasArraySize3_T);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&weightBiasArrayDev3_T), weightBiasArraySize3_T)");
  
    
  NUMBLOCKS = ((numberActivation3 * (numberActivation2))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  matrixTransposeSubBias<<< grid, threads >>>(weightBiasArrayDev3, weightBiasArrayDev3_T, numberActivation3, numberActivation2+1); 
  
    
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status,"  matrixTransposeAddBias<<< grid, threads >>>(weightBiasArrayDev3, weightBiasArrayDev3_T, numberActivation3, numberActivation2) ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  // copy data device memory to host:
  cudaMemcpy(weightBiasArray3_T, weightBiasArrayDev3_T, weightBiasArraySize3_T, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(weightBiasArray3_T, weightBiasArrayDev3_T, weightBiasArraySize3_T, cudaMemcpyDeviceToHost)  ");
  
  
   
  // updating with adam optimizer 
  
  float* mt3 = new float [numberActivation3 * (numberActivation2+1)];
  float* mt3_Dev;  
 
  float* vt3 = new float [numberActivation3 * (numberActivation2+1)];
  float* vt3_Dev;
   
  int len_dL_dW3 = numberActivation3 * (numberActivation2+1);
   
  memset(mt3, 0.0, dL_dW3_size);
  memset(vt3, 0.0, dL_dW3_size);
    
  status = cudaMalloc( (void **)(&mt3_Dev), dL_dW3_size);
  // checks for cuda errors  
  checkCudaErrors( status, " status = cudaMalloc( (void **)(&mt_Dev), dL_dW3_size);");
   
  status = cudaMalloc( (void **)(&vt3_Dev), dL_dW3_size);
  // checks for cuda errors  
  checkCudaErrors( status, " status = cudaMalloc( (void **)(&vt_Dev), dL_dW3_size);");
  

  float lr = 0.01;

  float beta1 = 0.9;

  float beta2 = 0.999;

  float epsilon = 1e-8;

  
  // copy data from host memory to the device:
   
  status = cudaMemcpy(mt3_Dev, mt3, dL_dW3_size, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(mt_Dev, mt, dL_dW3_size, cudaMemcpyHostToDevice );");

  
  status = cudaMemcpy(vt3_Dev, vt3, dL_dW3_size, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(vt_Dev, vt, dL_dW3_size, cudaMemcpyHostToDevice );");

    
  NUMBLOCKS = ((numberActivation3 * (numberActivation2+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;


  AdamOptUpdate<<< grid, threads >>>(weightBiasArrayDev3, dL_dW3_Dev, len_dL_dW3, lr, beta1, beta2, epsilon, mt3_Dev, vt3_Dev);

    
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status," AdamOptUpdate<<< grid, threads >>>(weightBiasArrayDev3, dL_dW3_Dev, len_dL_dW3, lr, beta1, beta2, epsilon, mt3_Dev, vt3_Dev) ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  

  // copy data device memory to host:
  cudaMemcpy(weightBiasArray3, weightBiasArrayDev3, weightBiasArraySize3, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );



  // copy data from host memory to the device:
   
  status = cudaMemcpy(mt3, mt3_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(mt3, mt3_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );");

  
  status = cudaMemcpy(vt3, vt3_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(vt, vt_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );");
  
    
  
  
  // reading pytorch W3 data after first backpropagation
  float* weightBiasArray3Pytorch = new float [numberActivation3 * (numberActivation2+1)];
  readfile("weightbias3_after_prop.txt", weightBiasArray3Pytorch);
  
  fprintf(stderr, " \n");
  fprintf(stderr, "Checking Pytorch layer 3 weight/bias match the CUDA implementation after the first back propagation: ");
  bool weightBiasArray3Match = true;
  for (int i = 0; i < numberActivation3; i++) {
       for (int j = 0; j <= numberActivation2; j++) {
         float torchval = weightBiasArray3Pytorch[IDX2C(i,j,(numberActivation2 + 1))];
         float cudaval = weightBiasArray3[IDX2C(i,j,(numberActivation2 + 1))];
         if ( (powf(torchval  - cudaval , 2.0)) > powf(0.01, 2.0) )  {
           weightBiasArray3Match =  false;
	   break;
	}
       }
  } 

  if (weightBiasArray3Match) {
     fprintf(stderr, "Yes, they match\n"); 
  } else {
     fprintf(stderr, "No, they do not match\n"); 
  }
  
  // calculation of dL_dZ2
  

  float *W3_dL_dZ3 = new float[numberActivation2  * numData];
  
  float *W3_dL_dZ3_Dev;

  int W3_dL_dZ3_DevSize = sizeof(float) * numberActivation2 * numData;
  

   
  NUMBLOCKS = ((numData * (numberActivation2))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  // allocate memory on the GPU;
  status = cudaMalloc( (void **)(&W3_dL_dZ3_Dev), W3_dL_dZ3_DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&W3_dL_dZ3_Dev), W3_dL_dZ3_DevSize); ");
  
  matrixMultRow<<< grid, threads >>>( weightBiasArrayDev3_T, dL_dZ3_Dev_T, W3_dL_dZ3_Dev, numberActivation2, numData, numberActivation3);

  status = cudaDeviceSynchronize( );
  
  checkCudaErrors( status,"matrixMulAddRowBasedARR2<<< grid, threads >>>( weightBiasArrayDev3_T, dL_dZ3_Dev, W3_dL_dZ3_Dev, (numberActivation2 + 1), numData, numberActivation3) ");
  
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  // copy data device memory to host:
  cudaMemcpy(W3_dL_dZ3, W3_dL_dZ3_Dev, W3_dL_dZ3_DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );

  
    
  float *dL_dZ2 = new float[numberActivation2  * numData];
  
  float *dL_dZ2_T = new float[numData  *  numberActivation2];

  float *dL_dZ2_Dev_T;
  
  float *dL_dZ2_Dev;

  int dL_dZ2_DevSize = sizeof(float) * numberActivation2 * numData;
   
  // allocate memory on the GPU;
  status = cudaMalloc( (void **)(&dL_dZ2_Dev), dL_dZ2_DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&dL_dZ2_Dev), dL_dZ2_DevSize); ");

  
  // allocate memory on the GPU;
  status = cudaMalloc( (void **)(&dL_dZ2_Dev_T), dL_dZ2_DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&dL_dZ2_Dev_T), dL_dZ2_DevSize); ");
  
  
  elementWiseMult<<< grid, threads >>>( diff_actValuesRelu2Dev_T, W3_dL_dZ3_Dev, dL_dZ2_Dev, numberActivation2 * numData);

  status = cudaDeviceSynchronize( );
  
  checkCudaErrors( status," elementWiseMult<<< grid, threads >>>(diff_actValuesRelu2_T, W3_dL_dZ3_Dev, dL_dZ2_Dev, numberActivation2 * numData); ");
  
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");   
  
  // copy data device memory to host:
  cudaMemcpy(dL_dZ2, dL_dZ2_Dev, dL_dZ2_DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );

 
  // Transpose of dL_dZ2 so it can be arranged in column major storage for matrix multiplication

  NUMBLOCKS = ((numData * (numberActivation2))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  matrixTransposeSubBias<<< grid, threads >>>(dL_dZ2_Dev, dL_dZ2_Dev_T, numberActivation2, numData); 

  // copy data device memory to host:
  cudaMemcpy(dL_dZ2_T, dL_dZ2_Dev_T, dL_dZ2_DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );
  
  
  

  
  // calculation of dL_dW2
    
  float* dL_dW2 = new float [numberActivation2 * (numberActivation+1)];
  
  
  float* dL_dW2_Dev;
  
  float* actValuesRelu1Dev;
  float* actValuesRelu1Dev_T;
  float* diff_actValuesRelu1Dev;
  float* diff_HactValuesRelu1;
  float* diff_actValuesRelu1Dev_T;
  
  float* diff_HactValuesRelu1_T;

  float* HactValuesRelu1_T = new float [numData * (numberActivation+1)];

  int actValuesDevSize1_T = sizeof(float) * numData * (numberActivation+1); 
  
  
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&actValuesRelu1Dev_T), actValuesDevSize1_T);
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMalloc( (void **)(&actValuesRelu1Dev_T), actValuesDevSize1_T); "); 

   
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&actValuesRelu1Dev), actValuesSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&actValuesRelu1Dev), actValuesDevSize); "); 
  
    


  // copy data from host memory to the device:
   
  status = cudaMemcpy(actValuesRelu1Dev, HactValuesRelu, actValuesSize, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status,"  cudaMemcpy(actValuesRelu1Dev, HactValuesRelu, actValuesDevSize, cudaMemcpyHostToDevice ); " );  
  
  
  NUMBLOCKS = ((numData * (numberActivation+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  
    
  // the column of the bias was filled with one
  matrixTransposeAddBias<<< grid, threads >>>(actValuesRelu1Dev, actValuesRelu1Dev_T, numData, numberActivation);    
  
  
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status,"  matrixTransposeAddBias<<< grid, threads >>>(actValuesRelu1Dev, actValuesRelu1Dev_T, numData, numberActivation)");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  
  // copy data device memory to host:
  cudaMemcpy(HactValuesRelu1_T, actValuesRelu1Dev_T,  actValuesDevSize1_T, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(HactValuesRelu_T, actValuesRelu2Dev_T,  yDataSize, cudaMemcpyDeviceToHost); " );   
 
  
  // this is to store the value of relu2 to be used for differentiation
  int diff_HactValuesRelu1DevSize = sizeof(float) * numData * (numberActivation);  
  
  diff_HactValuesRelu1 = new float [numData * numberActivation];
  
  diff_HactValuesRelu1_T = new float [numData * numberActivation];

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&diff_actValuesRelu1Dev),  diff_HactValuesRelu1DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&diff_actValuesRelu2Dev),  diff_HactValuesRelu2DevSize); "); 

  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&diff_actValuesRelu1Dev_T),  diff_HactValuesRelu1DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&diff_actValuesRelu2Dev),  diff_HactValuesRelu2DevSize); "); 


  
  status = cudaMemcpy(diff_actValuesRelu1Dev, actValuesRelu1Dev , diff_HactValuesRelu1DevSize, cudaMemcpyDeviceToDevice);
   
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMemcpy(diff_actValuesRelu1Dev, actValuesRelu1Dev , diff_HactValuesRelu1DevSize, cudaMemcpyDeviceToDevice); "); 
   // end of storing of the relu2 value
  
  NUMBLOCKS = ((numData * (numberActivation))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  matrixDiffReLu<<< grid, threads >>>(diff_actValuesRelu1Dev, (numData * numberActivation)); 
  
  // copy data device memory to host:
  cudaMemcpy(diff_HactValuesRelu1, diff_actValuesRelu1Dev,  diff_HactValuesRelu1DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, "cudaMemcpy(diff_HactValuesRelu1Dev, diff_actValuesRelu1Dev,  diff_HactValuesRelu1DevSize, cudaMemcpyDeviceToHost);");  
  
   
  matrixTranspose<<<grid, threads >>>(diff_actValuesRelu1Dev, diff_actValuesRelu1Dev_T, numData, numberActivation); 

  // copy data device memory to host:
  cudaMemcpy(diff_HactValuesRelu1_T, diff_actValuesRelu1Dev_T,  diff_HactValuesRelu1DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, "cudaMemcpy(diff_HactValuesRelu1Dev_T, diff_actValuesRelu1Dev_T,  diff_HactValuesRelu1DevSize, cudaMemcpyDeviceToHost);");  
 

   
  int dL_dW2_size =  (sizeof(float) * (numberActivation+1) * numberActivation2); 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&dL_dW2_Dev), dL_dW2_size);
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMalloc( (void **)(&dL_dW2_Dev), sizeof(float) * numberActivation2); "); 

    
  NUMBLOCKS = ((numberActivation2 * (numberActivation+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;
  
  matrixdL_dW3<<< grid, threads >>>(dL_dZ2_Dev, actValuesRelu1Dev_T, dL_dW2_Dev, numberActivation2, (numberActivation+1), numData);
  
  
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status,"  matrixdL_dW3<<< grid, threads >>>(actValuesDev3, actValuesRelu2Dev_T, dL_dW3_Dev, numberActivation3, (numberActivation2+1), numData)");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  // copy data device memory to host:
  cudaMemcpy(dL_dW2, dL_dW2_Dev,  dL_dW2_size, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(dL_dW2, dL_dW2_Dev,  dL_dW2_size, cudaMemcpyDeviceToHost);" ); 

    
  // Reading the dL_dW2 data
  float* dL_dW2_Pytorch = new float [numberActivation2 * (numberActivation+1)];
  // reading pytorch dL_dW3 data
  readfile("weightbias2_grad.txt", dL_dW2_Pytorch);
  bool dL_dW2_Match = true;
  fprintf(stderr, " \n");
  fprintf(stderr, "Checking dL_dW2 match between Pytorch and CUDA implementation: "); 
  for (int i = 0; i < numberActivation2; i++) {
      for (int j = 0; j <= numberActivation; j++) {
       float torchval = dL_dW2_Pytorch[IDX2C(i,j,(numberActivation + 1))];
       float cudaval = dL_dW2[IDX2C(i,j,(numberActivation + 1))];
       if ( (powf(torchval  - cudaval , 2.0)) > powf(0.01, 2.0) )  {
           dL_dW2_Match =  false;
	   break;
	}

      }
  }   
  if (dL_dW2_Match) {
     fprintf(stderr, "Yes, they match\n"); 
  } else {
     fprintf(stderr, "No, they do not match\n"); 
  }
  
   // transpose weightBiasArray2
      
   
  float* weightBiasArray2_T = new float [numberActivation2 * (numberActivation+1)];
  
  float* weightBiasArrayDev2_T;

  int weightBiasArraySize2_T = (sizeof(float) * numberActivation2 * (numberActivation+1));  
  
  // allocate memory on the GPU;
  status = cudaMalloc( (void **)(&weightBiasArrayDev2_T), weightBiasArraySize2_T);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&weightBiasArrayDev2_T), weightBiasArraySize2_T)");
  
    
  NUMBLOCKS = ((numberActivation2 * (numberActivation+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  matrixTransposeSubBias<<< grid, threads >>>(weightBiasArrayDev2, weightBiasArrayDev2_T, numberActivation2, numberActivation+1); 
  
    
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status,"  matrixTransposeAddBias<<< grid, threads >>>(weightBiasArrayDev2, weightBiasArrayDev2_T, numberActivation2, numberActivation) ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  // copy data device memory to host:
  cudaMemcpy(weightBiasArray2_T, weightBiasArrayDev2_T, weightBiasArraySize2_T, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(weightBiasArray2_T, weightBiasArrayDev2_T, weightBiasArraySize2_T, cudaMemcpyDeviceToHost)  ");

  
  // updating W2 with adam optimizer 
  
  float* mt2 = new float [numberActivation2 * (numberActivation+1)];
  float* mt2_Dev;  
 
  float* vt2 = new float [numberActivation2 * (numberActivation+1)];
  float* vt2_Dev;
   
  int len_dL_dW2 = numberActivation2 * (numberActivation+1);
   
  memset(mt2, 0.0, dL_dW2_size);
  memset(vt2, 0.0, dL_dW2_size);
    
  status = cudaMalloc( (void **)(&mt2_Dev), dL_dW2_size);
  // checks for cuda errors  
  checkCudaErrors( status, " status = cudaMalloc( (void **)(&mt2_Dev), dL_dW2_size);");
   
  status = cudaMalloc( (void **)(&vt2_Dev), dL_dW2_size);
  // checks for cuda errors  
  checkCudaErrors( status, " status = cudaMalloc( (void **)(&vt2_Dev), dL_dW2_size);");
  


  
  // copy data from host memory to the device:
   
  status = cudaMemcpy(mt2_Dev, mt2, dL_dW2_size, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(mt2_Dev, mt2, dL_dW2_size, cudaMemcpyHostToDevice );");

  
  status = cudaMemcpy(vt2_Dev, vt2, dL_dW2_size, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(vt_Dev, vt, dL_dW3_size, cudaMemcpyHostToDevice );");

    
  NUMBLOCKS = ((numberActivation2 * (numberActivation+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;


  AdamOptUpdate<<< grid, threads >>>(weightBiasArrayDev2, dL_dW2_Dev, len_dL_dW2, lr, beta1, beta2, epsilon, mt2_Dev, vt2_Dev);

    
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status," AdamOptUpdate<<< grid, threads >>>(weightBiasArrayDev2, dL_dW2_Dev, len_dL_dW2, lr, beta1, beta2, epsilon, mt2_Dev, vt2_Dev) ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  

  // copy data device memory to host:
  cudaMemcpy(weightBiasArray2, weightBiasArrayDev2, weightBiasArraySize2, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );



  // copy data from host memory to the device:
   
  status = cudaMemcpy(mt2, mt2_Dev, dL_dW2_size, cudaMemcpyDeviceToHost );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(mt3, mt3_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );");

  
  status = cudaMemcpy(vt2, vt2_Dev, dL_dW2_size, cudaMemcpyDeviceToHost );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(vt, vt_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );");
  
    
  
  // reading pytorch W2 data after first backpropagation
  float* weightBiasArray2Pytorch = new float [numberActivation2 * (numberActivation+1)];
  readfile("weightbias2_after_prop.txt", weightBiasArray2Pytorch);
  
  fprintf(stderr, " \n");
  fprintf(stderr, "Checking Pytorch layer 2 weight/bias match the CUDA implementation after the first back propagation: ");
  bool weightBiasArray2Match = true;
  
  for (int i = 0; i < numberActivation2; i++) {
     for (int j = 0; j <= numberActivation; j++) {
         float torchval = weightBiasArray2Pytorch[IDX2C(i,j,(numberActivation + 1))];
         float cudaval = weightBiasArray2[IDX2C(i,j,(numberActivation + 1))];
         if ( (powf(torchval  - cudaval , 2.0)) > powf(0.01, 2.0) )  {
           weightBiasArray2Match =  false;
	   break;
         }
     }

  } 
  if (weightBiasArray2Match) {
     fprintf(stderr, "Yes, they match\n"); 
  } else {
     fprintf(stderr, "No, they do not match\n"); 
  }

  
  // calculation of dL_dZ

    
  float *W2_dL_dZ2 = new float[numberActivation  * numData];
  
  float *W2_dL_dZ2_Dev;

  int W2_dL_dZ2_DevSize = sizeof(float) * numberActivation * numData;
  

   
  NUMBLOCKS = ((numData * (numberActivation))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  // allocate memory on the GPU;
  status = cudaMalloc( (void **)(&W2_dL_dZ2_Dev), W2_dL_dZ2_DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&W2_dL_dZ2_Dev), W2_dL_dZ2_DevSize); ");
  
  matrixMultRow<<< grid, threads >>>( weightBiasArrayDev2_T, dL_dZ2_Dev_T, W2_dL_dZ2_Dev, numberActivation, numData, numberActivation2);

  status = cudaDeviceSynchronize( );
  
  checkCudaErrors( status,"matrixMultRow<<< grid, threads >>>( weightBiasArrayDev2_T, dL_dZ2_Dev, W2_dL_dZ2_Dev, numberActivation, numData, numberActivation2);");
  
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  // copy data device memory to host:
  cudaMemcpy(W2_dL_dZ2, W2_dL_dZ2_Dev, W2_dL_dZ2_DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );

  
    
  float *dL_dZ = new float[numberActivation  * numData];
 
  float* dL_dZ_Dev; 
  
  int dL_dZ_DevSize = sizeof(float) * numberActivation * numData;
   
  // allocate memory on the GPU;
  status = cudaMalloc( (void **)(&dL_dZ_Dev), dL_dZ_DevSize);
  // checks for cuda errors  
  checkCudaErrors( status, " cudaMalloc( (void **)(&dL_dZ_Dev), dL_dZ_DevSize); ");

  
  
  elementWiseMult<<< grid, threads >>>( diff_actValuesRelu1Dev_T, W2_dL_dZ2_Dev, dL_dZ_Dev, numberActivation * numData);

  status = cudaDeviceSynchronize( );
  
  checkCudaErrors( status," elementWiseMult<<< grid, threads >>>(diff_actValuesRelu1_T, W2_dL_dZ2_Dev, dL_dZ_Dev, numberActivation * numData); ");
  
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");   
  
  // copy data device memory to host:
  cudaMemcpy(dL_dZ, dL_dZ_Dev, dL_dZ_DevSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );
   

  // Transpose of xxData input


  float* xxData_T = new float [numData * (numberOfFeatures+1)];

  int xxDataDevSize_T = sizeof(float) * numData * (numberOfFeatures+1); 
  
  float* xxDataDev_T;
  
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&xxDataDev_T), xxDataDevSize_T);
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMalloc( (void **)(&xxDataDev_T), xxDataDevSize_T); "); 

  
  NUMBLOCKS = ((numData * (numberOfFeatures+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

    
  // the column of the bias was filled with one
  matrixTransposeAddBias<<< grid, threads >>>(xxDataDev, xxDataDev_T, numData, numberOfFeatures);    
  
  
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status,"matrixTransposeAddBias<<< grid, threads >>>(xxData, xxDataDev_T, numData, numberOfFeatures);");    
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  
  // copy data device memory to host:
  cudaMemcpy(xxData_T, xxDataDev_T,  xxDataDevSize_T, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, "  " );


  // calculation of dL_dW
   
  
  float* dL_dW = new float [ numberOfFeatures * (numberActivation+1)];
  
  
  float* dL_dW_Dev;
  
  
  // copy data from host memory to the device:
   
  status = cudaMemcpy(actValuesRelu2Dev, HactValuesRelu2, actValuesDevSize2, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," cudaMemcpy(actValuesRelu2Dev_T, HactValuesRelu2, activationDevSize2, cudaMemcpyHostToDevice )" );  
   
  
  int dL_dW_size =  (sizeof(float) * (numberActivation) * (numberOfFeatures+1)); 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&dL_dW_Dev), dL_dW_size);
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMalloc( (void **)(&dL_dW_Dev), sizeof(float) * (numberActivation + 1) * numberOfFeatures); "); 

    
  NUMBLOCKS = ((numberOfFeatures * (numberActivation+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;
   
    
  matrixdL_dW3<<< grid, threads >>>(dL_dZ_Dev, xxDataDev_T, dL_dW_Dev, numberActivation, (numberOfFeatures+1), numData);
  
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status," matrixdL_dW3<<< grid, threads >>>(dL_dZ_Dev, xxDataDev_T, dL_dW2_Dev, numberActivation, (numberOfFeatures+1), numData);");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  
  // copy data device memory to host:
  cudaMemcpy(dL_dW, dL_dW_Dev,  dL_dW_size, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(dL_dW, dL_dW_Dev,  dL_dW_size, cudaMemcpyDeviceToHost);" ); 

  
  
  // Reading the dL_dW3 data
  float* dL_dW_Pytorch = new float [numberActivation * (numberOfFeatures+1)];
  // reading pytorch dL_dW3 data
  readfile("weightbias1_grad.txt", dL_dW_Pytorch);
  bool dL_dW_Match = true;
  fprintf(stderr, " \n");
  fprintf(stderr, "Checking dL_dW1 match between Pytorch and CUDA implementation: ");  
  for (int i = 0; i < numberActivation; i++) {
     for (int j = 0; j <= numberOfFeatures; j++) {
       float torchval = dL_dW_Pytorch[IDX2C(i,j,(numberOfFeatures + 1))];
       float cudaval = dL_dW[IDX2C(i,j,(numberOfFeatures + 1))];
       if ( (powf(torchval  - cudaval , 2.0)) > powf(0.01, 2.0) )  {
           dL_dW_Match =  false;
	   break;
       }	      
     }
  }
  
  if (dL_dW_Match) {
     fprintf(stderr, "Yes, they match\n"); 
  } else {
     fprintf(stderr, "No, they do not match\n"); 
  }
  
  // updating W2 with adam optimizer 
  
  float* mt = new float [numberActivation * (numberOfFeatures+1)];
  float* mt_Dev;  
 
  float* vt = new float [numberActivation * (numberOfFeatures+1)];
  float* vt_Dev;
   
  int len_dL_dW = numberActivation * (numberOfFeatures+1);
   
  memset(mt, 0.0, dL_dW_size);
  memset(vt, 0.0, dL_dW_size);
    
  status = cudaMalloc( (void **)(&mt_Dev), dL_dW_size);
  // checks for cuda errors  
  checkCudaErrors( status, " status = cudaMalloc( (void **)(&mt_Dev), dL_dW_size);");
   
  status = cudaMalloc( (void **)(&vt_Dev), dL_dW_size);
  // checks for cuda errors  
  checkCudaErrors( status, " status = cudaMalloc( (void **)(&vt_Dev), dL_dW_size);");
  


  
  // copy data from host memory to the device:
   
  status = cudaMemcpy(mt_Dev, mt, dL_dW_size, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(mt_Dev, mt, dL_dW_size, cudaMemcpyHostToDevice );");

  
  status = cudaMemcpy(vt_Dev, vt, dL_dW_size, cudaMemcpyHostToDevice );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(vt_Dev, vt, dL_dW_size, cudaMemcpyHostToDevice );");

    
  NUMBLOCKS = ((numberActivation * (numberOfFeatures+1))+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;


  AdamOptUpdate<<< grid, threads >>>(weightBiasArrayDev, dL_dW_Dev, len_dL_dW, lr, beta1, beta2, epsilon, mt_Dev, vt_Dev);

    
  status = cudaDeviceSynchronize( );
   
  checkCudaErrors( status," AdamOptUpdate<<< grid, threads >>>(weightBiasArrayDev, dL_dW_Dev, len_dL_dW, lr, beta1, beta2, epsilon, mt_Dev, vt_Dev) ");
 
  status = cudaGetLastError(); 
  
  checkCudaErrors( status,"cudaGetLastError()");  
  

  // copy data device memory to host:
  cudaMemcpy(weightBiasArray, weightBiasArrayDev, weightBiasArraySize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status );



  // copy data from host memory to the device:
   
  status = cudaMemcpy(mt, mt_Dev, dL_dW_size, cudaMemcpyDeviceToHost );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(mt3, mt3_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );");

  
  status = cudaMemcpy(vt, vt_Dev, dL_dW_size, cudaMemcpyDeviceToHost );
  // checks for cuda errors
  checkCudaErrors( status," status = cudaMemcpy(vt, vt_Dev, dL_dW3_size, cudaMemcpyDeviceToHost );");
  
  
  // reading pytorch W1 data after first backpropagation
  float* weightBiasArrayPytorch = new float [numberActivation * (numberOfFeatures + 1)];
  readfile("weightbias1_after_prop.txt", weightBiasArrayPytorch);
  fprintf(stderr, " \n");
  fprintf(stderr, "Checking Pytorch layer 1 weight/bias match the CUDA implementation after the first back propagation: ");
  bool weightBiasArrayMatch = true;  
  for (int i = 0; i < numberActivation; i++) {
       for (int j = 0; j <= numberOfFeatures; j++) {
         float torchval = weightBiasArrayPytorch[IDX2C(i,j,(numberOfFeatures + 1))];
         float cudaval = weightBiasArray[IDX2C(i,j,(numberOfFeatures + 1))];
         if ( (powf(torchval  - cudaval , 2.0)) > powf(0.01, 2.0) )  {
           weightBiasArray2Match =  false;
	   break;
         }
       }
  }
  
   
  if (weightBiasArrayMatch) {
     fprintf(stderr, "Yes, they match\n"); 
  } else {
     fprintf(stderr, "No, they do not match\n"); 
  }











  //matrixTranspose(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols) 
  

  // the column of the bias was filled with one
  //matrixTransposeAddBias<<< grid, threads >>>(actValuesRelu2Dev, actValuesRelu2Dev_T, numData, numberActivation2);    

  cudaFree( weightBiasArrayDev );
  cudaFree( weightBiasArrayDev2 );
  cudaFree( weightBiasArrayDev3 );
  cudaFree( xxDataDev ); 
  cudaFree( xxDataDev_T ); 
  cudaFree( actValuesDev ); 
  cudaFree( yDataDev ); 
    
  cudaFree(dL_dZ3_Dev);
  cudaFree(dL_dZ2_Dev);
  cudaFree(dL_dZ_Dev);
  cudaFree(dL_dZ3_Dev_T);  
  cudaFree(dL_dZ2_Dev_T);  
  
  cudaFree(dL_dW3_Dev);
  cudaFree(dL_dW2_Dev);
  cudaFree(dL_dW_Dev);

  cudaFree(W3_dL_dZ3_Dev);
  cudaFree(W2_dL_dZ2_Dev);

  
  cudaFree(diff_actValuesRelu2Dev_T); 
  cudaFree(diff_actValuesRelu2Dev);  
  cudaFree(diff_actValuesRelu1Dev_T); 
  cudaFree(diff_actValuesRelu1Dev); 
  
  cudaFree(mt_Dev);
  cudaFree(mt2_Dev);
  cudaFree(mt3_Dev);
  

  cudaFree(vt_Dev);
  cudaFree(vt2_Dev);
  cudaFree(vt3_Dev);
  


  delete[] weightBiasArray;
  delete[] weightBiasArray2;
  delete[] weightBiasArray3;
  delete[] xxData;
  delete[] xxData_T;
  delete[] HactValues;
  delete[] HactValues2;
  delete[] HactValues3;
  delete[] dL_dZ3;
  delete[] dL_dZ2;
  delete[] dL_dZ;
  delete[] dL_dZ3_T;  
  delete[] dL_dZ2_T;  

  delete[] dL_dW3;
  delete[] dL_dW2;
  delete[] dL_dW;
  
  delete[] W3_dL_dZ3;
  delete[] W2_dL_dZ2;

  delete[] diff_HactValuesRelu2_T; 
  delete[] diff_HactValuesRelu2; 
  delete[] diff_HactValuesRelu1_T; 
  delete[] diff_HactValuesRelu1; 

  delete[] mt;
  delete[] mt2;
  delete[] mt3;
  

  delete[] vt;
  delete[] vt2;
  delete[] vt3;

  return 0;
};	
