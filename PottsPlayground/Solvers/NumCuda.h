#ifndef NumCudaHeader
#define NumCudaHeader

#define PY_SSIZE_T_CLEAN //why? not sure.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <stdio.h>
#include <numpy/ndarrayobject.h>
#include <typeinfo>
#include <atomic>
#include <stdint.h>

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define __h__ __host__
    #define __d__ __device__ 
#else
    #define __h__
    #define __d__
#endif

template <typename T> class NumCuda{
public:
	T * dd; //device data
	T * hd; //host data
    std::atomic_int * dd_refcount; //flags to indicate if the descructor should free the memory at dd or hd when the object is deleted
    std::atomic_int * hd_refcount;
	npy_intp dims[5]; //5 dimensions max. Making this fixed length so that I don't have to track memory allocation
	int ndims;
	int nBytes;
    int nElements;
	npy_intp NpyTypeId;

    // ================================================================================= CONSTRUCTORS, DESTRUCTORS, ETC
    __h__ __d__ NumCuda(const NumCuda && source){ //needed with some compilers (gcc 9) but not others (gcc 11)
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

    __h__ __d__ NumCuda(const NumCuda & source){
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

    __h__ __d__ NumCuda& operator=(const NumCuda & source){
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

    __h__ __d__ ~NumCuda();

	//class constructor to create a whole new array
	__h__ NumCuda(int d0, int d1=-1, int d2=-1, int d3=-1, int d4=-1);

        
    __h__ __d__ NumCuda(){
        hd_refcount = NULL;
        dd_refcount = NULL; //since no memory is allocated here
    }

	//class constructor to create a NumCuda from an existing Numpy object
	__h__ NumCuda(PyArrayObject* source, int ndims_, bool CopyToHost, bool copyToDevice){
        construct_from_pyarray(source, ndims_, CopyToHost, copyToDevice);
    }

    //extracts numpy array from a python dictionary or class object first:
    __h__ NumCuda(PyObject* obj, const char* name, int ndims_, bool CopyToHost, bool copyToDevice){
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


    __h__ void construct_from_pyarray(PyArrayObject* source, int ndims_, bool CopyToHost, bool copyToDevice);

    // ========================================================================================= SYNCHRONIZATION FUNCTIONS
	__h__ void CopyHostToDevice();
	__h__ void CopyDeviceToHost();

    // ======================================================================================================Export and display

	__h__ PyObject* ExportNumpy(){
        T* copied_data = (T*)malloc(nBytes);
        memcpy(copied_data, hd, nBytes);
        // printf("Size of intp: %i\n", sizeof(npy_intp));
        npy_intp * copied_dims = (npy_intp*)malloc(sizeof(npy_intp)*ndims);
        for (int i = 0; i<ndims;i++)
            copied_dims[i] = dims[i];
        // printf("NpyTypeId: %i, %i, %i\n", NpyTypeId, NPY_INT32, NPY_FLOAT32);
        PyObject* x = PyArray_SimpleNewFromData(npy_intp(ndims), copied_dims, NpyTypeId, copied_data);
        return x;
    }

    __h__ __d__ void display(){
        //prints out the matrix to std out.
        
    }

    // ========================================================================================================== ARRAY ACCESS
    __h__ __d__ inline T& operator ()(int indx1){
        #ifdef __CUDA_ARCH__
            // printf("Here\n");
            return dd[indx1];
        #else
            // printf("There\n");
            return hd[indx1];
        #endif
    }

    __h__ __d__ inline T& operator ()(int indx1, int indx2){
        #ifdef __CUDA_ARCH__
            return dd[indx1*dims[1] + indx2];
        #else
            return hd[indx1*dims[1] + indx2];
        #endif
    }

    __h__ __d__ inline T& operator ()(int indx1, int indx2, int indx3){
        #ifdef __CUDA_ARCH__
            return dd[indx1*dims[1]*dims[2] + indx2*dims[2] + indx3];
        #else
            return hd[indx1*dims[1]*dims[2] + indx2*dims[2] + indx3];
        #endif
    }

    __h__ __d__ inline T& operator ()(int indx1, int indx2, int indx3, int indx4){
        #ifdef __CUDA_ARCH__
            return dd[indx1*dims[1]*dims[2]*dims[3] + indx2*dims[2]*dims[3] + indx3*dims[3] + indx4];
        #else
            return hd[indx1*dims[1]*dims[2]*dims[3] + indx2*dims[2]*dims[3] + indx3*dims[3] + indx4];
        #endif
    }

    // ===================================================================================================== MATH OPERATIONS
    __h__ __d__ int ArgMin(){
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

    __h__ __d__ T min(){
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

    __h__ __d__ float mean(){
        float m = sum();
        return m/nElements;
    }

    __h__ __d__ T sum(){
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

//functions that depend on __CUDACC__ have to be defined outside the class def,
//otherwise they are "inline" and the extern keyword does squat
// ========================================================================================= SYNCHRONIZATION FUNCTIONS
template <typename T> __h__ void NumCuda<T>::CopyHostToDevice(){
    #ifdef __CUDACC__
        cudaError_t err = cudaSuccess;
        err = cudaMemcpy(dd, hd, nBytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("Error copying to device memory: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
            throw;
        }
    #else
        printf("Cannot copy to device, NumCuda was not compiled with GPU support\n");
    #endif
}
    
template <typename T> __h__ void NumCuda<T>::CopyDeviceToHost(){
    #ifdef __CUDACC__
        cudaError_t err = cudaSuccess;
        err = cudaMemcpy(hd, dd, nBytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("Error copying from device memory: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
            throw;
        }
    #else
        printf("Cannot copy from device, NumCuda was not compiled with GPU support\n");
    #endif
}

template <typename T> __h__ void NumCuda<T>::construct_from_pyarray(PyArrayObject* source, int ndims_, bool CopyToHost, bool copyToDevice){
    if (ndims_ < 0) ndims_ = PyArray_NDIM(source);
    ndims = ndims_;
    if (PyArray_NDIM(source) != ndims){
        printf("Numpy array should have %i dimensions, not %i\n", ndims, PyArray_NDIM(source));
        PyErr_SetString(PyExc_ValueError, "Number of array dimensions is not expected");
        throw;
    }

    //type checking:
    if (typeid(T) == typeid(int32_t)) NpyTypeId = NPY_INT32;
    else if (typeid(T) == typeid(float)) NpyTypeId = NPY_FLOAT32;
    if (PyArray_TYPE(source) != NpyTypeId){
        printf("A Numpy array type does not match required C object type\n");
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
        hd_refcount = new std::atomic_int;
        *hd_refcount = 1;
    }
    else {
        hd = (T*)PyArray_DATA(source);
        hd_refcount = NULL; //not our job, since the data is owned by Python
    }

    //set up device array memory
    #ifdef __CUDACC__
        cudaError_t err = cudaSuccess;
        err = cudaMalloc((void **)&dd, nBytes);
        dd_refcount = new std::atomic_int;
        *dd_refcount = 1;
        if (err != cudaSuccess) {
            printf("Error Allocating Cuda memory: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
            throw;
        }
        if (copyToDevice) CopyHostToDevice();
    #else
        dd_refcount = NULL;
    #endif
    
}

//class constructor to create a whole new array
template <typename T> __h__ NumCuda<T>::NumCuda(int d0, int d1, int d2, int d3, int d4){
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

    nBytes = nElements*sizeof(T);

    //set up host array memory
    hd = (T*)malloc(nBytes);
    hd_refcount = new std::atomic_int;// (int*)malloc(sizeof(std::atomic_int));
    *hd_refcount = 1;

    //set up device array memory
    #ifdef __CUDACC__
        cudaError_t err = cudaMalloc((void **)&dd, nBytes);
        dd_refcount = new std::atomic_int;
        *dd_refcount = 1;
        if (err != cudaSuccess) {
            printf("Error Allocating Cuda memory: %s.  Array size = [%i, %i, %i, %i, %i]\n", cudaGetErrorString(err), int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3]), int(dims[4]));
            PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
            throw;
        }
    #else
        dd_refcount = NULL;
    #endif

    if (typeid(T) == typeid(int32_t)) NpyTypeId = NPY_INT32;
    else if (typeid(T) == typeid(float)) NpyTypeId = NPY_FLOAT32;
}

template <typename T> __h__ __d__ NumCuda<T>::~NumCuda(){
    // printf("NumCuda host pointer at deletion time: %i\n", hd);
    #ifdef __CUDA_ARCH__
        return;
    #else

        if (hd_refcount != NULL){
            *hd_refcount -= 1;
            if (*hd_refcount == 0){
                free(hd);
                free(hd_refcount);
            }
        }

        #ifdef __CUDACC__
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
        #endif
    #endif
}
//explicity create and compile in either GpuCore or GpuCoreAlt, so that the right nvcc stuff is present 
extern template class NumCuda<int>;
extern template class NumCuda<float>;

#endif //NumCuda include guard