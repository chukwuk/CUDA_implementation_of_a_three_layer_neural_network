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
  float* dL_dZ3 = new float [numData];
  float* dL_dZ3_Dev; 

  BLOCKSIZE = 758;
  NUMBLOCKS = ((numData)+BLOCKSIZE-1)/BLOCKSIZE;
  

  int yDataSize = (sizeof(float) * numData) ;
   
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
  
  status = cudaMemcpy(dL_dZ3_Dev, actValuesDev3, yDataSize, cudaMemcpyDeviceToDevice);
  checkCudaErrors( status, "  cudaMemcpy(dL_dZ3_Dev, actValuesDev3, yDataSize, cudaMemcpyDeviceToDevice);" );

  // copy data device memory to host:
  cudaMemcpy(dL_dZ3, actValuesDev3,  yDataSize, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(dL_dZ3, actValuesDev3,  yDataSize, cudaMemcpyDeviceToHost);  " );
  
  
    
    
  fprintf(stderr, " \n"); 
  fprintf(stderr, " Output of the dL_dZ \n"); 
  for (int i = 0; i < numData; i++) {
      fprintf(stderr,"%f ", dL_dZ3[i]); 
  }
  
  fprintf(stderr, " \n");
  
  
  // dL/dW3 USING ADAM optimizer at a learning rate of 0.001

  float* dL_dW3 = new float [numberActivation3 * (numberActivation2+1)];

  
  float* dL_dW3_Dev;
  
  float* actValuesRelu2Dev;
  float* actValuesRelu2Dev_T;
  
  float* HactValuesRelu2_T = new float [numData * numberActivation2+1];



  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&actValuesRelu2Dev_T), actValuesDevSize2);
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
 

 //__global__  void matrixTranspose(float* matrixArray, float* matrixArrayTranpose, int nrows, int ncols)  
  
  
  NUMBLOCKS = ((numData * numberActivation2)+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;

  fprintf(stderr, " Output of the relu2 \n");
  for (int i = 0; i < numData; i++) {
      for (int j = 0; j < numberActivation2; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu2[IDX2C(i,j,numberActivation2)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
  

  matrixTranspose<<< grid, threads >>>(actValuesRelu2Dev, actValuesRelu2Dev_T, numData, numberActivation2);    
  
  // copy data device memory to host:
  cudaMemcpy(HactValuesRelu2_T, actValuesRelu2Dev_T,  actValuesDevSize2, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(HactValuesRelu_T, actValuesRelu2Dev_T,  yDataSize, cudaMemcpyDeviceToHost); " );

  
   
  fprintf(stderr, " Output of the Transpose relu2 \n");
  for (int i = 0; i < numberActivation2; i++) {
      for (int j = 0; j < numData; j++) { 
	fprintf(stderr,"%f ", HactValuesRelu2_T[IDX2C(i,j,numData)]);
 
      }
      fprintf(stderr, " \n");
  }   
  
  fprintf(stderr, " \n");
  
  int dL_dW3_size =  (sizeof(float) * (numberActivation2+1) * numberActivation3); 
  //allocate memory on the GPU device
  status = cudaMalloc( (void **)(&dL_dW3_Dev), dL_dW3_size);
  // checks for cuda errors  
  checkCudaErrors( status, "  cudaMalloc( (void **)(&dL_dW3_Dev), sizeof(float) * numberActivation2); "); 

    
  NUMBLOCKS = ((numberActivation3 * numberActivation2)+BLOCKSIZE-1)/BLOCKSIZE;
  grid.x = NUMBLOCKS;
  
  matrixdL_dW3<<< grid, threads >>>(actValuesDev3, actValuesRelu2Dev_T, dL_dW3_Dev, numberActivation3, numberActivation2, numData);
  

  
  // copy data device memory to host:
  cudaMemcpy(dL_dW3, dL_dW3_Dev,  dL_dW3_size, cudaMemcpyDeviceToHost);  
  // checks for cuda errors
  checkCudaErrors( status, " cudaMemcpy(dL_dW3, dL_dW3_Dev,  dL_dW3_size, cudaMemcpyDeviceToHost);" ); 

  
  fprintf(stderr, " \n");
 
  fprintf(stderr, " Output of the dL_dW3 \n"); 
  for (int i = 0; i < numberActivation2; i++) {
      fprintf(stderr,"%f ", dL_dW3[i]); 
  }
  
  fprintf(stderr, " \n");
  

  

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
