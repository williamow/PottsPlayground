#include <stdio.h>
#include <cuda_runtime.h>
#include <typeinfo>


//a template for easing the (already easy) translations between numpy arrays from python and procesing in Cuda.

template <typename T> class NumCuda{
public:
	T * dd; //device data
	T * hd; //host data
	int dims[5]; //5 dimensions max. Making this fixed length so that I don't have to track memory allocation
	int ndims;
	int nBytes;
    int nElements;
	int NpyTypeId;

	//class constructor to create a whole new array
	__host__ NumCuda(int ndims_, int* dims_){
        //set up metadata:
        this->ndims = ndims_;
        this->nBytes = sizeof(T);
        for (int i = 0; i<this->ndims; i++) {
            this->dims[i] = dims_[i];
            this->nBytes *= dims_[i];
        }

        //set up host array memory
        this->hd = (T*)malloc(this->nBytes);

        //set up device array memory
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&this->dd, this->nBytes);
        if (err != cudaSuccess) {
            printf("Error Allocating Cuda memory: %s\n", cudaGetErrorString(err));
            // PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
            throw;
        }
    }

    __host__ __device__ ~NumCuda(){
    	#ifdef __CUDA_ARCH__
            return;
        #else
            cudaFree(dd);
        #endif
        printf("Freeing memory, just like I should\n");
        // free(hd); //I guess this gets handled automatically? I get a complaint of "already freed"
    }

    // __device__ ~NumCuda(){
    	// return;
    // }

	__host__ void CopyHostToDevice(){
        cudaError_t err = cudaSuccess;
        err = cudaMemcpy(dd, hd, nBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying to device memory: %s\n", cudaGetErrorString(err));
            // PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
            throw;
        }
    }


	__host__ void CopyDeviceToHost(){
        cudaError_t err = cudaSuccess;
        err = cudaMemcpy(hd, dd, nBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Error copying from device memory: %s\n", cudaGetErrorString(err));
            // PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
            throw;
        }
    }


    __host__ __device__ inline T& operator ()(int indx1, int indx2){
        #ifdef __CUDA_ARCH__
            return this->dd[indx1*this->dims[1] + indx2];
        #else
            return this->hd[indx1*this->dims[1] + indx2];
        #endif
    }

};

//=======================================================================================================
__global__ void buggy(
    NumCuda<int> kmap 
    )
{
    printf("A kmap value: %i\n", kmap(1, 2));
    kmap(1, 2) = 136;
    return;
}

int main(){
	cudaError_t err = cudaSuccess;

    int dims[2] = {100, 100};
    NumCuda<int> kmap(2, dims);

    kmap(1, 2) = 200;
    kmap.CopyHostToDevice();

    for (int i = 0; i<1; i++){
	    // Launch the kernel
	    int threadsPerBlock = 2;
	    int blocksPerGrid = 2;//(nNHPPs + threadsPerBlock - 1) / threadsPerBlock;
	    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);


    	cudaSetupArgument(&kmap,sizeof(kmap),0);
    	cudaConfigureCall(blocksPerGrid,threadsPerBlock);
	    cudaLaunch(buggy);

	    // buggy<<<blocksPerGrid, threadsPerBlock>>>(kmap);
	    cudaDeviceSynchronize(); //wait for GPU processing to finish
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("kernel execution error: %s\n", cudaGetErrorString(err));
            // PyErr_SetString(PyExc_ValueError, "CUDA kernel execution error");
            // return NULL;
        }
        printf("Kmap value from C-code: %i\n", kmap(1, 2));
        // printf("")
        kmap.CopyDeviceToHost();
        printf("Kmap value from C-code: %i\n", kmap(1, 2));
	}


 


    return 0;
}
