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

int
main( int argc, char* argv[ ] )
{ 
  //srand(time(0));
//  fprintf (stderr, "Amount of data transfered to the device is %lld GB\n", bytes4euc/1000000000);
  
  vector<char*> array;
  int numberActivation = 5;
  int numberActivation2 = 3;
  int numberActivation3 = 1;
  int numberOfFeatures = 10;
  int numData = 2;   
   
  int wRows = numberActivation;
  int xCols = numData;
  int wColsXRows = numberOfFeatures;
  
  
  int BLOCKSIZE = 758;
  int NUMBLOCKS = ((wRows * numData)+BLOCKSIZE-1)/BLOCKSIZE;
    
  fprintf (stderr, "NUMBER OF BLOCKS is %d\n", NUMBLOCKS);
  
  
  // checking memory coalescing
   
  float* xxData = new float[numData * numberOfFeatures];
  
  for (int i = 0; i < (numData * numberOfFeatures); i++) {
      xxData[i] = (rand() % 8)/1.0; 
  }
  
 
  float* yData = new float[numData];
  
  float* weightBiasArray = new float[numberActivation * (numberOfFeatures+1)];
  float* weightBiasArray2 = new float[(numberActivation + 1) * numberActivation2];
  float* weightBiasArray3 = new float[(numberActivation2 + 1) * numberActivation3];
  for (int i = 0; i < numberActivation; i++) {
       for (int j = 0; j <= numberOfFeatures; j++) {
	   weightBiasArray[IDX2C(i,j,(numberOfFeatures+1))] = (rand() % 9)/1.0;
       }
  }
  

  FILE *file;
  char line[256]; // Buffer to store each line
  char *token;

  file = fopen("weightbias.txt", "r"); // Replace "example.txt" with your file name

  if (file == NULL) {
     fprintf(stderr, "Error opening file");
     return 1;
  }
  int count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      // Remove trailing newline character if present
      line[strcspn(line, "\n")] = 0; 

      // Split the line by spaces
      token = strtok(line, " ");
      while (token != NULL) {
          //fprintf(stderr, "%s ", token);
          float t = atof(token);
          weightBiasArray[count] = t;

	  array.push_back(token);
	  token = strtok(NULL, " ");
          //fprintf(stderr, "%d ", count);
	  count+=1;        
      }

      //fprintf(stderr, " \n");
    }

  fclose(file);
  //int count;
  
  fprintf(stderr, " \n");
  
  fprintf(stderr, " %d", count);


  fprintf(stderr, " \n");
  
  int cnt = 0; 
  for (int i = 0; i < numberActivation; i++) {
       for (int j = 0; j <= numberOfFeatures; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }

  
  fprintf(stderr, " \n");
  
  file = fopen("weightbias2.txt", "r"); // Replace "example.txt" with your file name

  if (file == NULL) {
     fprintf(stderr, "Error opening file");
     return 1;
  }
  count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      // Remove trailing newline character if present
      line[strcspn(line, "\n")] = 0; 

      // Split the line by spaces
      token = strtok(line, " ");
      while (token != NULL) {
          //fprintf(stderr, "%s ", token);
          float t = atof(token);
          weightBiasArray2[count] = t;

	 // array.push_back(token);
	  token = strtok(NULL, " ");
          //fprintf(stderr, "%d ", count);
	  count+=1;        
      }

      //fprintf(stderr, " \n");
    }

  fclose(file);
  
  
  cnt = 0; 
  for (int i = 0; i < numberActivation2; i++) {
       for (int j = 0; j <= numberActivation; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray2[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");

  } 
  fprintf(stderr, " \n");
      
  
  
  file = fopen("weightbias3.txt", "r"); // Replace "example.txt" with your file name

  if (file == NULL) {
     fprintf(stderr, "Error opening file");
     return 1;
  }
  count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      // Remove trailing newline character if present
      line[strcspn(line, "\n")] = 0; 

      // Split the line by spaces
      token = strtok(line, " ");
      while (token != NULL) {
          //fprintf(stderr, "%s ", token);
          float t = atof(token);
          weightBiasArray3[count] = t;

	 // array.push_back(token);
	  token = strtok(NULL, " ");
          //fprintf(stderr, "%d ", count);
	  count+=1;        
      }

      //fprintf(stderr, " \n");
    }

  fclose(file);
  

  fprintf(stderr, " \n");
   
  
  
  cnt = 0; 
  for (int i = 0; i < numberActivation3; i++) {
       for (int j = 0; j <= numberActivation2; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray3[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");

  file = fopen("inputdata.txt", "r"); // Replace "example.txt" with your file name

  if (file == NULL) {
     fprintf(stderr, "Error opening file");
     return 1;
  }
  count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      // Remove trailing newline character if present
      line[strcspn(line, "\n")] = 0; 

      // Split the line by spaces
      token = strtok(line, " ");
      while (token != NULL) {
          //fprintf(stderr, "%s ", token);
          float t = atof(token);
          xxData[count] = t;

	 // array.push_back(token);
	  token = strtok(NULL, " ");
          //fprintf(stderr, "%d ", count);
	  count+=1;        
      }

      //fprintf(stderr, " \n");
    }

  fclose(file);
  

  fprintf(stderr, " \n");
  
  
  cnt = 0; 
  for (int i = 0; i < numData; i++) {
       for (int j = 0; j < numberOfFeatures; j++) {
          fprintf(stderr, "%.4f ", xxData[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
   
    
  fprintf(stderr, " \n");
  
  
  file = fopen("targetdata.txt", "r"); // Replace "example.txt" with your file name

  if (file == NULL) {
     fprintf(stderr, "Error opening file");
     return 1;
  }
  count = 0;
  while (fgets(line, sizeof(line), file) != NULL) {
      // Remove trailing newline character if present
      line[strcspn(line, "\n")] = 0; 

      // Split the line by spaces
      token = strtok(line, " ");
      while (token != NULL) {
          //fprintf(stderr, "%s ", token);
          float t = atof(token);
          yData[count] = t;

	 // array.push_back(token);
	  token = strtok(NULL, " ");
          //fprintf(stderr, "%d ", count);
	  count+=1;        
      }

      //fprintf(stderr, " \n");
    }

  fclose(file);
  

  fprintf(stderr, " \n");
  
  
  cnt = 0; 
  for (int i = 0; i < numData; i++) {
      fprintf(stderr, "%.4f  ", yData[cnt]);
      cnt+=1;
  }
   
    
  fprintf(stderr, " \n");
  



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
  
  fprintf(stderr, " Output of the linear1 \n"); 
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < wRows; j++) { 
	fprintf(stderr,"%f ", HactValues[IDX2C(i,j,wRows)]);
 
        //fprintf(stderr,"%f ", HactValues[IDX2C(i,j,wRows)]);
      //printf("   activation values: %f \n", HactValues[(numData * wRows)-1]);
      }
      fprintf(stderr, " \n");
  }  
  
  
  fprintf(stderr, " \n");

  
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


  
  fprintf(stderr, " Output of the relu1 \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < wRows; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu[IDX2C(i,j,wRows)]);
 
        //fprintf(stderr,"%f ", HactValues[IDX2C(i,j,wRows)]);
      //printf("   activation values: %f \n", HactValues[(numData * wRows)-1]);
      }
      fprintf(stderr, " \n");
  }  
    
  fprintf(stderr, " \n");

  
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


  
  fprintf(stderr, " Output of the linear2 \n"); 
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation2; j++) { 
	fprintf(stderr,"%f ", HactValues2[IDX2C(i,j,numberActivation2)]);
 
        //fprintf(stderr,"%f ", HactValues[IDX2C(i,j,wRows)]);
      //printf("   activation values: %f \n", HactValues[(numData * wRows)-1]);
      }
      fprintf(stderr, " \n");
  }  
  
  fprintf(stderr, " \n");
    
  
  
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


  
  fprintf(stderr, " Output of the relu2 \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation2; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu2[IDX2C(i,j,numberActivation2)]);
 
        //fprintf(stderr,"%f ", HactValues[IDX2C(i,j,wRows)]);
      //printf("   activation values: %f \n", HactValues[(numData * wRows)-1]);
      }
      fprintf(stderr, " \n");
  }  
  
  
  
  fprintf(stderr, " \n");
  
   

    
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


  
  fprintf(stderr, " Output of the linear3 \n"); 
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation3; j++) { 
	fprintf(stderr,"%f ", HactValues3[IDX2C(i,j,numberActivation3)]);
 
        //fprintf(stderr,"%f ", HactValues[IDX2C(i,j,wRows)]);
      //printf("   activation values: %f \n", HactValues[(numData * wRows)-1]);
      }
      fprintf(stderr, " \n");
  }  
  
  fprintf(stderr, " \n");

 
  
  
  
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
  


  fprintf(stderr, " Output of the Sigmoid \n"); 
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation3; j++) { 
	fprintf(stderr,"%f ", HactValues3[IDX2C(i,j,numberActivation3)]);
 
        //fprintf(stderr,"%f ", HactValues[IDX2C(i,j,wRows)]);
      //printf("   activation values: %f \n", HactValues[(numData * wRows)-1]);
      }
      fprintf(stderr, " \n");
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
  
  
    
    
  fprintf(stderr, " \n"); 
  fprintf(stderr, " Output of the dL_dZ3 \n"); 
  for (int i = 0; i < numberActivation3; i++) {
    for (int j = 0; j < numData; j++) {
        fprintf(stderr,"%f ", dL_dZ3[IDX2C(i,j,numData)]);	
    }

    fprintf(stderr, " \n");
  }
  fprintf(stderr, " \n");
  
  
  
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
  
  
  fprintf(stderr, " \n"); 
  fprintf(stderr, " Output of the Transpose of dL_dZ3 \n"); 
  for (int i = 0; i < numData; i++) {
    for (int j = 0; j < numberActivation3; j++) {
        fprintf(stderr,"%f ", dL_dZ3_T[IDX2C(i,j,numberActivation3)]); 
    }

    fprintf(stderr, " \n");
  }
  fprintf(stderr, " \n");
  

  
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

  fprintf(stderr, " Output of the relu2 \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation2; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu2[IDX2C(i,j,numberActivation2)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
  
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

  
   
  fprintf(stderr, " Output of the Transpose relu2 \n");
  for (int i = 0; i <= numberActivation2; i++) {
      for (int j = 0; j < numData; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu2_T[IDX2C(i,j,numData)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
   
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
  
  
  fprintf(stderr, " Output of the differentiation of relu2 output \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j <  numberActivation2; j++) { 
	fprintf(stderr,"%f ", diff_HactValuesRelu2[IDX2C(i,j,numberActivation2)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
   
  
    
  fprintf(stderr, " Output of the differentiation of relu2 output  after Transpose\n");
  for (int i = 0; i < numberActivation2; i++) {
      for (int j = 0; j <  numData; j++) { 
	fprintf(stderr,"%f ", diff_HactValuesRelu2_T[IDX2C(i,j,numData)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
   


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

  
  fprintf(stderr, " \n");
 
  fprintf(stderr, " Output of the dL_dW3 \n"); 
  for (int i = 0; i <= numberActivation2; i++) {
      fprintf(stderr,"%f ", dL_dW3[i]); 
  }
  
  fprintf(stderr, " \n");
  
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
  
  
  fprintf(stderr, " \n");
  fprintf(stderr, " The output of the Transpose of weight bias array 3\n");
  cnt = 0; 
  for (int i = 0; i < numberActivation2; i++) {
       for (int j = 0; j < numberActivation3; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray3_T[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");

   
   
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
  
    
  fprintf(stderr, " \n");
  fprintf(stderr, " The new weight and bias of layer 3 after the first back propagation\n");
  cnt = 0; 
  for (int i = 0; i < numberActivation3; i++) {
       for (int j = 0; j <= numberActivation2; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray3[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");

  
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

  
  fprintf(stderr, " \n");
  fprintf(stderr, " The output of transpose of W3 multiplied by dL_dZ3 \n");
  cnt = 0; 
  for (int i = 0; i < numberActivation2; i++) {
       for (int j = 0; j < numData; j++) {
          fprintf(stderr, "%.4f ", W3_dL_dZ3[IDX2C(i,j,numData)]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");
  
    
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
  
  
  
  fprintf(stderr, " \n");
  fprintf(stderr, " The output of dL_dZ2 \n");
  cnt = 0; 
  for (int i = 0; i < numberActivation2; i++) {
       for (int j = 0; j < numData; j++) {
          fprintf(stderr, "%.4f ", dL_dZ2[IDX2C(i,j,numData)]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");

  
  
  fprintf(stderr, " \n");
  fprintf(stderr, " The output of the tranpose of dL_dZ2 \n");
  cnt = 0; 
  for (int i = 0; i < numData; i++) {
       for (int j = 0; j < numberActivation2; j++) {
          fprintf(stderr, "%.4f ", dL_dZ2_T[IDX2C(i,j,numberActivation2)]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");


  
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

  fprintf(stderr, " Output of the relu1 \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu[IDX2C(i,j,numberActivation)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
  
    
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

  
   
  fprintf(stderr, " Output of the Transpose relu1 \n");
  for (int i = 0; i <= numberActivation; i++) {
      for (int j = 0; j < numData; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu1_T[IDX2C(i,j,numData)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
  
 
  
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
  
  
  fprintf(stderr, " Output of the differentiation of relu1 output \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j <  numberActivation; j++) { 
	fprintf(stderr,"%f ", diff_HactValuesRelu1[IDX2C(i,j,numberActivation)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");

  
  fprintf(stderr, " Output of the differentiation of relu1 output  after Transpose\n");
  for (int i = 0; i < numberActivation; i++) {
      for (int j = 0; j <  numData; j++) { 
	fprintf(stderr,"%f ", diff_HactValuesRelu1_T[IDX2C(i,j,numData)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
   
  

   
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

  
  fprintf(stderr, " \n");
 
  fprintf(stderr, " Output of the dL_dW2 \n");

  for (int i = 0; i < numberActivation2; i++) {
      for (int j = 0; j <= numberActivation; j++) {
       fprintf(stderr,"%f ", dL_dW2[IDX2C(i,j,(numberActivation+1))]);
      }
     fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");

  
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
  
  
  fprintf(stderr, " \n");
  fprintf(stderr, " The output of the Transpose of weight bias array 2\n");
  cnt = 0; 
  for (int i = 0; i < numberActivation; i++) {
       for (int j = 0; j < numberActivation2; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray2_T[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");




  
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
  
    
  fprintf(stderr, " \n");
  fprintf(stderr, " The new weight and bias of layer 2 after the first back propagation\n");
  cnt = 0; 
  for (int i = 0; i < numberActivation2; i++) {
       for (int j = 0; j <= numberActivation; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray2[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");


  
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

  
  fprintf(stderr, " \n");
  fprintf(stderr, " The output of transpose of W2 multiplied by dL_dZ2 \n");
  cnt = 0; 
  for (int i = 0; i < numberActivation; i++) {
       for (int j = 0; j < numData; j++) {
          fprintf(stderr, "%.4f ", W2_dL_dZ2[IDX2C(i,j,numData)]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");
  
    
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
  
  
  
  fprintf(stderr, " \n");
  fprintf(stderr, " The output of dL_dZ \n");
  cnt = 0; 
  for (int i = 0; i < numberActivation; i++) {
       for (int j = 0; j < numData; j++) {
          fprintf(stderr, "%.4f ", dL_dZ[IDX2C(i,j,numData)]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");

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

  fprintf(stderr, " Output of the xxData \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberOfFeatures; j++) { 
	fprintf(stderr,"%f ", xxData[IDX2C(i,j,numberOfFeatures)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
  
    
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

  
   
  fprintf(stderr, " Output of the xxData Tranpose \n");
  for (int i = 0; i <= numberOfFeatures; i++) {
      for (int j = 0; j < numData; j++) { 
	fprintf(stderr,"%f ", xxData_T[IDX2C(i,j,numData)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
  


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

  
  
  

  fprintf(stderr, " \n");
 
  fprintf(stderr, " Output of the dL_dW \n");

  for (int i = 0; i < numberActivation; i++) {
      for (int j = 0; j <= numberOfFeatures; j++) {
       fprintf(stderr,"%f ", dL_dW[IDX2C(i,j,(numberOfFeatures+1))]);
      }
     fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");

  
  
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
  
    
  fprintf(stderr, " \n");
  fprintf(stderr, " The new weight and bias of layer 2 after the first back propagation\n");
  cnt = 0; 
  for (int i = 0; i < numberActivation; i++) {
       for (int j = 0; j <= numberOfFeatures; j++) {
          fprintf(stderr, "%.4f ", weightBiasArray[cnt]);
	  cnt+=1;
       }

      fprintf(stderr, " \n");
  }
  
  fprintf(stderr, " \n");











  //matrixTranspose(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols) 
  

  // the column of the bias was filled with one
  //matrixTransposeAddBias<<< grid, threads >>>(actValuesRelu2Dev, actValuesRelu2Dev_T, numData, numberActivation2);    

  cudaFree( weightBiasArrayDev );
  cudaFree( xxDataDev ); 
  cudaFree( actValuesDev ); 
  
  cudaFree( yDataDev ); 
  
  delete[] weightBiasArray;
  
  delete[] weightBiasArray2;
  delete[] weightBiasArray3;
  delete[] xxData;
  delete[] HactValues;
  
  delete[] HactValues2;

  delete[] HactValues3;
   
  delete[] dL_dZ3;

  return 0;
};	
