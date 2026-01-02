

#ifndef __MATRIXMUL_H
#define __MATRIXMUL_H


__global__  void matrixKaimingUniformInitilization(float* activation, int actLength, int inputFeatures);

__global__  void matrixMulAddColBasedARR(float* weightBias, float* xData,  float* activationValues, int wRows, int xCols, int wColsXRows);

__global__  void matrixMulAddRowBasedARR(float* weightBias, float* xData,  float* activationValues, int wRows, int xCols, int wColsXRows);

__global__  void matrixMulColBasedARR(double* weight, double* bias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wRowsXRows);


__global__  void matrixMulRowBasedARR(double* weight, double* bias, double* xData, double* activationValues, long long int wRows, long long int xCols, long long int wRowsXRows);

__global__  void matrixMulAddRowBasedARR2(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows);

__global__  void matrixReLu(float* activation, int actLength);


__global__  void matrixSigmoid(float* activation, int actLength);

__global__  void elementWiseSub(float* firstArray, float* secondArray, int arraySize); 

__global__  void matrixTranspose(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols); 

__global__ void matrixTransposeAddBias(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols); 

__global__  void matrixdL_dW3(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows); 


__global__  void AdamOptUpdate(float* weightBias, float* dL_dW3, int len, float lr, float beta1, float beta2, float epsilon, float* mt, float* vt);

__global__  void matrixTransposeSubBias(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols);


__global__  void matrixDiffReLu(float* activation, int actLength);


__global__  void matrixMultRow(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows);


__global__  void elementWiseMult(float* firstArray, float* secondArray, float* outputArray, int arraySize);

#endif
