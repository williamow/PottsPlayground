#define PY_SSIZE_T_CLEAN //why? not sure.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <stdio.h>
#include <numpy/ndarrayobject.h>
#include <cuda_runtime.h>
#include <typeinfo>


//a template for easing the (already easy) translations between numpy arrays from python and procesing in Cuda.

template <typename T> class NumCuda{
public:
	T * dd; //device data
	T * hd; //host data
    bool ddDel; //flags to indicate if the descructor should free the memory at dd or hd when the object is deleted
    bool hdDel;
	int dims[5]; //5 dimensions max. Making this fixed length so that I don't have to track memory allocation
	int ndims;
	int nBytes;
    int nElements;
	int NpyTypeId;

    // ================================================================================= CONSTRUCTORS, DESTRUCTORS, ETC
    __host__ NumCuda(NumCuda & source){
        //custom copy constructor,
        //so that we can keep track of how many references there are to the data arrays,
        //so that we can correctly decide when to free the allocated memory
        dd = source.dd;
        hd = source.dd;
        ddDel = false; 
        hdDel = false;
        //a better way would be to make a linked list out of the objects,
        //and only free memory when deleting the last object in the list.
        //however, since this is so far only needed for a specific use case, this suffices.

        //and regular copying for everything else:
        for (int i = 0; i < 5; i++) dims[i] = source.dims[i];
        ndims = source.ndims;
        nBytes = source.nBytes;
        nElements = source.nElements;
        NpyTypeId = source.NpyTypeId;
    }

    __host__ ~NumCuda(){
        if (ddDel) cudaFree(dd);
        if (hdDel) free(hd);
    }


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
        hdDel = true;

        //set up device array memory
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&this->dd, this->nBytes);
        ddDel = true;
        if (err != cudaSuccess) {
            printf("Error Allocating Cuda memory: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
            throw;
        }
    }

	//class constructor to create a NumCuda from an existing Numpy object
	__host__ NumCuda(PyArrayObject* source, int ndims_, bool CopyToHost, bool copyToDevice){
        ndims = ndims_;
        if (PyArray_NDIM(source) != ndims){
            printf("Numpy array should have %i dimensions, not %i\n", ndims, PyArray_NDIM(source));
            PyErr_SetString(PyExc_ValueError, "Number of array dimensions is not expected");
            throw;
        }

        //type checking:
        if (typeid(T) == typeid(int)) NpyTypeId = NPY_INT;
        else if (typeid(T) == typeid(float)) NpyTypeId = NPY_FLOAT;
        if (PyArray_TYPE(source) != NpyTypeId){
            PyErr_SetString(PyExc_ValueError, "Numpy array type does not match C object type");
            throw;
        }

        //transfer metadata:
        nBytes = sizeof(T);
        for (int i = 0; i<ndims; i++) {
            dims[i] = PyArray_SHAPE(source)[i];
            nBytes *= dims[i];
        }

        //set up host array memory
        if (CopyToHost){
            hd = (T*)malloc(nBytes);
            memcpy(hd, PyArray_DATA(source), nBytes);
            hdDel = true;
        }
        else {
            hd = (T*)PyArray_DATA(source);
            hdDel = false;
        }

        //set up device array memory
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&dd, nBytes);
        ddDel = true;
        if (err != cudaSuccess) {
            printf("Error Allocating Cuda memory: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
            throw;
        }
        if (copyToDevice) CopyHostToDevice();
    }

    // ========================================================================================= SYNCHRONIZATION FUNCTIONS
	__host__ void CopyHostToDevice(){
        cudaError_t err = cudaSuccess;
        err = cudaMemcpy(dd, hd, nBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying to device memory: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
            throw;
        }
    }

	__host__ void CopyDeviceToHost(){
        cudaError_t err = cudaSuccess;
        err = cudaMemcpy(hd, dd, nBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Error copying from device memory: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
            throw;
        }
    }

	__host__ PyArrayObject* ExportNumpy(){
        T* copied_data = (T*)malloc(nBytes);
        memcpy(copied_data, hd, nBytes);
        return PyArray_SimpleNewFromData(npy_intp(ndims), (npy_intp*)dims, npy_intp(NpyTypeId), copied_data);
    }

    // ========================================================================================================== ARRAY ACCESS
    __host__ __device__ inline T& operator ()(int indx1){
        #ifdef __CUDA_ARCH__
            return dd[indx1];
        #else
            return hd[indx1];
        #endif
    }

    __host__ __device__ inline T& operator ()(int indx1, int indx2){
        #ifdef __CUDA_ARCH__
            return dd[indx1*dims[1] + indx2];
        #else
            return hd[indx1*dims[1] + indx2];
        #endif
    }

    __host__ __device__ inline T& operator ()(int indx1, int indx2, int indx3){
        #ifdef __CUDA_ARCH__
            return dd[indx1*dims[1]*dims[2] + indx2*dims[2] + indx3];
        #else
            return hd[indx1*dims[1]*dims[2] + indx2*dims[2] + indx3];
        #endif
    }

    // ===================================================================================================== MATH OPERATIONS
    __host__ __device__ int ArgMin(){
        //returns the index of the minimum value, without indexing
        T * data;
        #ifdef __CUDA_ARCH__
            data =  dd;
        #else
            data = hd;
        #endif
        T min = data[0];
        int min_index = 0;
        for (int i = 0; i<nElements; i++){
            if (data[i] < min) {
                min = data[i];
                min_index = i;
            }
        }
        return i;
    }

};