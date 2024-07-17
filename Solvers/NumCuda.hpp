#ifndef NumCudaHeader
#define NumCudaHeader


#define PY_SSIZE_T_CLEAN //why? not sure.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <stdio.h>
#include <numpy/ndarrayobject.h>
#include <cuda_runtime.h>
#include <typeinfo>

template <typename T> class NumCuda{
public:
	T * dd; //device data
	T * hd; //host data
    int * dd_refcount; //flags to indicate if the descructor should free the memory at dd or hd when the object is deleted
    int * hd_refcount;
	npy_intp dims[5]; //5 dimensions max. Making this fixed length so that I don't have to track memory allocation
	int ndims;
	int nBytes;
    int nElements;
	npy_intp NpyTypeId;

    // ================================================================================= CONSTRUCTORS, DESTRUCTORS, ETC
    __host__ __device__ NumCuda(NumCuda && source){ //needed with some compilers (gcc 9) but not others (gcc 11)
        dd = source.dd;
        hd = source.hd;

        //only do reference counting on host side - device side will never fully "own" the array:
        #ifndef __CUDA_ARCH__
            // printf("Starting reference counting\n");
            dd_refcount = source.dd_refcount;
            if (dd_refcount != NULL) *dd_refcount += 1;
            hd_refcount = source.hd_refcount;
            if (hd_refcount != NULL) *hd_refcount += 1;
            // printf("Done updating reference counts\n");
        #endif
        
        //and regular copying for everything else:
        for (int i = 0; i < 5; i++) dims[i] = source.dims[i];
        ndims = source.ndims;
        nBytes = source.nBytes;
        nElements = source.nElements;
        NpyTypeId = source.NpyTypeId;
    }

    __host__ __device__ NumCuda(NumCuda & source){
        //custom copy constructor,
        //so that we can keep track of how many references there are to the data arrays,
        //so that we can correctly decide when to free the allocated memory
        // printf("Making a copy!\n");
        // #ifdef NumCudaDebug {printf("In copy constructor, source hd=%i, dest hd=%i\n", source.hd, hd);}
        dd = source.dd;
        hd = source.hd;

        //only do reference counting on host side - device side will never fully "own" the array:
        #ifndef __CUDA_ARCH__
            // printf("Starting reference counting\n");
            dd_refcount = source.dd_refcount;
            if (dd_refcount != NULL) *dd_refcount += 1;
            hd_refcount = source.hd_refcount;
            if (hd_refcount != NULL) *hd_refcount += 1;
            // printf("Done updating reference counts\n");
        #endif
        
        //and regular copying for everything else:
        for (int i = 0; i < 5; i++) dims[i] = source.dims[i];
        ndims = source.ndims;
        nBytes = source.nBytes;
        nElements = source.nElements;
        NpyTypeId = source.NpyTypeId;
    }

    __host__ __device__ NumCuda& operator=(const NumCuda & source){
        //ugh.  So maybe what I really needed this whole time was an assigment operator, not the copy constructor.
        // printf("In assignment operator, source hd=%i, dest hd=%i\n", source.hd, hd);
        this->~NumCuda(); //destruct existing data

        

        dd = source.dd;
        hd = source.hd;
        
        //only do reference counting on host side - device side will never fully "own" the array:
        #ifndef __CUDA_ARCH__
            // printf("Starting reference counting\n");
            dd_refcount = source.dd_refcount;
            if (dd_refcount != NULL) *dd_refcount += 1;
            hd_refcount = source.hd_refcount;
            if (hd_refcount != NULL) *hd_refcount += 1;
            // printf("Done updating reference counts\n");
        #endif

        //and regular copying for everything else:
        for (int i = 0; i < 5; i++) dims[i] = source.dims[i];
        ndims = source.ndims;
        nBytes = source.nBytes;
        nElements = source.nElements;
        NpyTypeId = source.NpyTypeId;
        return *this;
    }

    __host__ __device__ ~NumCuda(){
        // printf("NumCuda host pointer at deletion time: %i\n", hd);
        #ifdef __CUDA_ARCH__
            return;
        #else
            if (dd_refcount != NULL){
                *dd_refcount -= 1;
                if (*dd_refcount == 0){
                    cudaError_t err = cudaFree(dd);
                    if (err != cudaSuccess) {
                        printf("Error Freeing Cuda memory: %s\n", cudaGetErrorString(err));
                        PyErr_SetString(PyExc_ValueError, "cudaFree error");
                    }
                    free(dd_refcount);
                }
            }

            if (hd_refcount != NULL){
                *hd_refcount -= 1;
                if (*hd_refcount == 0){
                    free(hd);
                    free(hd_refcount);
                }
            }
            
        #endif
        
    }

	//class constructor to create a whole new array
	__host__ NumCuda(int d0, int d1=-1, int d2=-1, int d3=-1, int d4=-1){
        dims[0] = d0;
        dims[1] = d1;
        dims[2] = d2;
        dims[3] = d3;
        dims[4] = d4;

        nElements = 1;
        for (ndims = 0; ndims < 5; ndims++){
            if (dims[ndims] == -1) break;
            nElements *= dims[ndims];
        }

        //set up metadata:
        // if (d1==-1 && d2==-1 && d3==-1 && d4==-1) ndims=1;
        // else if (d2==-1 && d3==-1 && d4==-1) ndims=2;
        // else if (d3==-1 && d4==-1) ndims=3;
        // else if (d4==-1) ndims=4;
        // else ndims=5;
        // // ndims = ndims_;
        // nElements = d0*d1*d2*d3*d4;
        

        nBytes = nElements*sizeof(T);

        //set up host array memory
        hd = (T*)malloc(nBytes);
        hd_refcount = (int*)malloc(sizeof(int));
        *hd_refcount = 1;

        //set up device array memory
        cudaError_t err = cudaMalloc((void **)&dd, nBytes);
        dd_refcount = (int*)malloc(sizeof(int));
        *dd_refcount = 1;
        if (err != cudaSuccess) {
            printf("Error Allocating Cuda memory: %s.  Array size = [%i, %i, %i, %i, %i]\n", cudaGetErrorString(err), int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4]));
            PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
            throw;
        }
        if (typeid(T) == typeid(int)) NpyTypeId = NPY_INT;
        else if (typeid(T) == typeid(float)) NpyTypeId = NPY_FLOAT;
    }

        
    __host__ __device__ NumCuda(){
        hd_refcount = NULL;
        dd_refcount = NULL; //since no memory is allocated here
    }

	//class constructor to create a NumCuda from an existing Numpy object
	__host__ NumCuda(PyArrayObject* source, int ndims_, bool CopyToHost, bool copyToDevice){
        construct_from_pyarray(source, ndims_, CopyToHost, copyToDevice);
    }

    //extracts numpy array from a python dictionary or class object first:
    __host__ NumCuda(PyObject* obj, const char* name, int ndims_, bool CopyToHost, bool copyToDevice){
        PyArrayObject* source;
        if(PyDict_Check(obj)){
            //dictionary retrieval:
            if (!PyDict_Contains(obj, PyUnicode_FromString(name))) printf("Error, object does not have attribute %s\n", name);
            source = (PyArrayObject*)PyDict_GetItemString(obj, name);
        }
        else{
            if (!PyObject_HasAttr(obj, PyUnicode_FromString(name))) printf("Error, object does not have attribute %s\n", name);
            source = (PyArrayObject*)PyObject_GetAttr(obj, PyUnicode_FromString(name));
        }
        construct_from_pyarray(source, ndims_, CopyToHost, copyToDevice);
    }


    __host__ void construct_from_pyarray(PyArrayObject* source, int ndims_, bool CopyToHost, bool copyToDevice){
        if (ndims_ < 0) ndims_ = PyArray_NDIM(source);
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
        nElements = 1;
        for (int i = 0; i<ndims; i++) {
            dims[i] = PyArray_SHAPE(source)[i];
            nElements *= dims[i];
        }
        nBytes = nElements*sizeof(T);

        //set up host array memory
        if (CopyToHost){
            hd = (T*)malloc(nBytes);
            memcpy(hd, PyArray_DATA(source), nBytes);
            hd_refcount = (int*)malloc(sizeof(int));
            *hd_refcount = 1;
        }
        else {
            hd = (T*)PyArray_DATA(source);
            hd_refcount = NULL; //not our job, since the data is owned by Python
        }

        //set up device array memory
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&dd, nBytes);
        dd_refcount = (int*)malloc(sizeof(int));
        *dd_refcount = 1;
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

    // ======================================================================================================Export and display

	__host__ PyObject* ExportNumpy(){
        // printf("Here 1\n");
        T* copied_data = (T*)malloc(nBytes);
        // printf("Here 2\n");
        memcpy(copied_data, hd, nBytes);
        // printf("Here 3\n");
        // printf("Size of intp: %i\n", sizeof(npy_intp));
        npy_intp * copied_dims = (npy_intp*)malloc(sizeof(npy_intp)*ndims);
        for (int i = 0; i<ndims;i++)
            copied_dims[i] = dims[i];
        // printf("NpyTypeId: %i, %i, %i\n", NpyTypeId, NPY_INT, NPY_FLOAT);
        PyObject* x = PyArray_SimpleNewFromData(npy_intp(ndims), copied_dims, NpyTypeId, copied_data);
        // printf("Here 4\n");
        return x;
    }

    __host__ __device__ void display(){
        //prints out the matrix to std out.
        
    }

    // ========================================================================================================== ARRAY ACCESS
    __host__ __device__ inline T& operator ()(int indx1){
        #ifdef __CUDA_ARCH__
            // printf("Here\n");
            return dd[indx1];
        #else
            // printf("There\n");
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

    __host__ __device__ inline T& operator ()(int indx1, int indx2, int indx3, int indx4){
        #ifdef __CUDA_ARCH__
            return dd[indx1*dims[1]*dims[2]*dims[3] + indx2*dims[2]*dims[3] + indx3*dims[3] + indx4];
        #else
            return hd[indx1*dims[1]*dims[2]*dims[3] + indx2*dims[2]*dims[3] + indx3*dims[3] + indx4];
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
        return min_index;
    }

    __host__ __device__ T min(){
        //returns the minimum value
        T * data;
        #ifdef __CUDA_ARCH__
            data =  dd;
        #else
            data = hd;
        #endif
        T min = data[0];
        for (int i = 0; i<nElements; i++){
            if (data[i] < min) {
                min = data[i];
            }
        }
        return min;
    }

    __host__ __device__ float mean(){
        //returns the minimum value
        T * data;
        #ifdef __CUDA_ARCH__
            data =  dd;
        #else
            data = hd;
        #endif
        float m = 0;
        // printf("nElements: %i\n", nElements);
        for (int i = 0; i<nElements; i++)
            m += data[i];
        return m/nElements;
    }

    __host__ __device__ T sum(){
        //returns the minimum value
        T * data;
        #ifdef __CUDA_ARCH__
            data =  dd;
        #else
            data = hd;
        #endif
        T m = 0;
        // printf("nElements: %i\n", nElements);
        for (int i = 0; i<nElements; i++)
            m += data[i];
        return m;
    }

};

#endif