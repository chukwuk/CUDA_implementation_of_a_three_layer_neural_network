#include <math.h>
#include "matrixmul.h"    
#include <stdio.h>

using namespace std;


__global__  void matrixMulColBasedARR(double* weight, double* bias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
   //  int x_index = gid/NUMDATAS;
   //  int y_index = gid % NUMDATAS;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
	long long int index = gid / xCols;
        long long int indexW = index * wColsXRows;
        long long int indexStart = gid % wColsXRows;
        long long int IndexMul = indexStart * wColsXRows; 
    	double  sum = 0.0;
	for (long long int i = 0; i < wColsXRows; i++)  {
	   sum+=(weight[i+indexW] * xData[i+IndexMul]);
           
       	}
	sum+=bias[index];
        activationValues[gid] = sum;	
	 
    }

}


__global__  void matrixMulRowBasedARR(double* weight, double* bias, double* xData,  double* activationValues, long long int wRows, long long int xCols, long long int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   long long int gid = (long long int) blockIdx.x * (long long int) blockDim.x + (long long int) threadIdx.x;
   //  int x_index = gid/NUMDATAS;
   //  int y_index = gid % NUMDATAS;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
	long long int index = gid / xCols;
	long long int indexR = gid % xCols;
        long long int indexW = index * wColsXRows;
        long long int indexStart = gid % wColsXRows;
        long long int IndexMul = indexStart * wColsXRows; 
    	double  sum = 0.0;
	for (long long int i = 0; i < wColsXRows; i++)  {
	   sum+=(weight[i+indexW] * xData[i+IndexMul]);
           
       	}
	sum+=bias[index];
        activationValues[index+(indexR*wRows)] = sum;	
	 
    }

}



__global__  void matrixMulAddColBasedARR(float* weightBias, float* xData,  float* activationValues, int wRows, int xCols, int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   int gid =  blockIdx.x *  blockDim.x + threadIdx.x;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
        int index = gid / xCols;
        int indexW = index * (wColsXRows+1);
        int indexStart = gid % xCols;
        int IndexMul = indexStart * wColsXRows; 
    	float sum = 0.0;
	for (int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexW] * xData[i+IndexMul]);
           
       	}
	sum+=weightBias[indexW+wColsXRows];
        activationValues[gid] = sum;	
	 
    }

}


__global__  void matrixMulAddRowBasedARR(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
	int index = gid / xCols;
	int indexR = gid - (index * xCols);
        int indexW = index * (wColsXRows+1);
       // int indexStart = gid - (index * xCols);//gid % xCols;
        int IndexMul = indexR * wColsXRows; 
    	float sum = 0.0;
	for (int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexW] * xData[i+IndexMul]);
           
       	}
	sum+=weightBias[indexW+wColsXRows];
        activationValues[index+(indexR*wRows)] = sum;	
	 
    }

}

// This is matrix multiplication for getting the transpose of the matrix output
__global__  void matrixMulAddRowBasedARR2(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    
   // euclideanDistance[gid] = ((float)gid)*((float)NUMDATA);
    if (gid < (wRows * xCols)) {
        	
	int index = gid / wRows;
	int indexR = gid - (index * wRows);
        int indexW = index * (wColsXRows);
        //int indexStart = gid - (index * xCols);//gid % xCols;
        int indexMul = indexR * (wColsXRows+1); 
    	float sum = 0.0;
	for (int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexMul] * xData[i+indexW]);
           
       	}
	sum+=weightBias[indexMul+wColsXRows];
        activationValues[gid] = sum;	
	 
    }

}



__global__  void matrixReLu(float* activation, int actLength) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;

    if (gid < actLength) {
       activation[gid] = (activation[gid] > 0.0) ? activation[gid] : 0.0;
    }
}


__global__  void matrixDiffReLu(float* activation, int actLength) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;

    if (gid < actLength) {
       activation[gid] = (activation[gid] > 0.0) ? 1.0 : 0.0;
    }
}



__global__  void matrixSigmoid(float* activation, int actLength) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;

    if (gid < actLength) {
       activation[gid] = 1/(1 + exp(-activation[gid]));
    }
}


__global__  void elementWiseSub(float* firstArray, float* secondArray, int arraySize) {

   
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (gid < (arraySize)) {
	firstArray[gid]=firstArray[gid] - secondArray[gid]; 
   }

}


__global__  void elementWiseMult(float* firstArray, float* secondArray, float* outputArray, int arraySize) {

   
   int gid = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (gid < (arraySize)) {
	outputArray[gid]=firstArray[gid] * secondArray[gid]; 
   }

}



__global__  void matrixTranspose(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    int index = gid / ncols;
    int indexR = gid - (index * ncols);

    if (gid < (nrows*ncols)) {
       matrixArrayTranspose[index+(indexR*nrows)] = matrixArray[gid];
    }
}


__global__  void matrixTransposeAddBias(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    int index = gid / ncols;
    int indexR = gid - (index * ncols);

    if (gid < (nrows*(ncols+1))) {
       if (gid < (nrows*ncols)) { 
          matrixArrayTranspose[index+(indexR*nrows)] = matrixArray[gid];
       } else {
          matrixArrayTranspose[gid] = 1.0;
       }
       
    } 
}


__global__  void matrixTransposeSubBias(float* matrixArray, float* matrixArrayTranspose, int nrows, int ncols) {
     
    int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    int index = gid / ncols;
    int indexR = gid - (index * ncols);

    if (gid < (nrows*ncols)) {
       matrixArrayTranspose[index+(indexR*nrows)] = matrixArray[gid];
    } 
}



__global__  void matrixdL_dW3(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    
    
    if (gid < (wRows * xCols)) {
        	
        int index = gid / xCols;
        int indexW = index * (wColsXRows);
        int indexStart = gid % xCols;
        int IndexMul = indexStart * wColsXRows; 
    	float sum = 0.0;
	for (int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexW] * xData[i+IndexMul]);
           
       	}
        activationValues[gid] = sum/float(wColsXRows);	
	 
    }

}


__global__  void matrixdL_dW2(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    
    
    if (gid < (wRows * xCols)) {
        	
        int index = gid / xCols;
        int indexW = index * (wColsXRows);
        int indexStart = gid % xCols;
        int IndexMul = indexStart * wColsXRows; 
    	float sum = 0.0;
	for (int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexW] * xData[i+IndexMul]);
           
       	}
        activationValues[gid] = sum/float(wColsXRows);	
	 
    }

}





__global__  void AdamOptUpdate(float* weightBias, float* dL_dW3, int len, float lr, float beta1, float beta2, float epsilon, float* mt, float* vt) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    
    
    if (gid < len) {
        
        mt[gid] = (beta1*mt[gid]) + (1 - beta1)*dL_dW3[gid];	
        vt[gid] = (beta2*vt[gid]) + (1 - beta2)*(powf(dL_dW3[gid], 2.0));

	float mt_crt = mt[gid]/(1-beta1);
	float vt_crt = vt[gid]/(1-beta2);
	
	weightBias[gid] = weightBias[gid] - ((mt_crt/(sqrt(vt_crt + epsilon)))*lr);
         	
    }

}



__global__  void matrixMultRow(float* weightBias, float* xData,  float* activationValues, int wRows,  int xCols, int wColsXRows) {


   // int blockNum = blockIdx.y*gridDim.x + blockIdx.x;
   // int blockThreads = blockNum*blockDim.x*blockDim.y;
   // int gid = blockThreads + threadIdx.y*blockDim.x + threadIdx.x;
   int gid =  blockIdx.x *  blockDim.x +  threadIdx.x;
    
    
    if (gid < (wRows * xCols)) {
        	
        int index = gid / xCols;
        int indexW = index * (wColsXRows);
        int indexStart = gid % xCols;
        int IndexMul = indexStart * wColsXRows; 
    	float sum = 0.0;
	for (int i = 0; i < wColsXRows; i++)  {
	   sum+=(weightBias[i+indexW] * xData[i+IndexMul]);
           
       	}
        activationValues[gid] = sum;	
	 
    }

}
