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
	int dims[5]; //5 dimensions max. Making this fixed length so that I don't have to track memory allocation
	int ndims;
	int nBytes;
	int NpyTypeId;

	//class constructor to create a whole new array
	NumCuda(int ndims, int *dims);

	//class constructor to create a NumCuda from an existing Numpy object
	NumCuda(PyArrayObject* source, int ndims, bool CopyToHost, bool copyToDevice);

	void CopyHostToDevice();
	void CopyDeviceToHost();

	PyArrayObject* ExportNumpy();

};

template <typename T> NumCuda<T>::NumCuda(int ndims_, int* dims_){
    //set up metadata:
	ndims = ndims_;
    nBytes = sizeof(T);
    for (int i = 0; i<ndims; i++) {
    	dims[i] = dims_[i];
    	nBytes *= dims[i];
    }

    //set up host array memory
    hd = (T*)malloc(nBytes);

    //set up device array memory
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&dd, nBytes);
    if (err != cudaSuccess) {
        printf("Error Allocating Cuda memory: %s\n", cudaGetErrorString(err));
        PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
    }
}



template <typename T> NumCuda<T>::NumCuda(PyArrayObject* source, int ndims_, bool CopyToHost, bool copyToDevice){
	ndims = ndims_;
    if (PyArray_NDIM(source) != ndims){
    	printf("Numpy array should have %i dimensions, not %i\n", ndims, PyArray_NDIM(source));
    	PyErr_SetString(PyExc_ValueError, "Number of array dimensions is not expected");
    }

    //type checking:
    if (typeid(T) == typeid(int)) NpyTypeId = NPY_INT;
    else if (typeid(T) == typeid(float)) NpyTypeId = NPY_FLOAT;
    if (PyArray_TYPE(source) != NpyTypeId){
    	PyErr_SetString(PyExc_ValueError, "Numpy array type does not match C object type");
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
    }
    else hd = (T*)PyArray_DATA(source);

    //set up device array memory
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&dd, nBytes);
    if (err != cudaSuccess) {
        printf("Error Allocating Cuda memory: %s\n", cudaGetErrorString(err));
        PyErr_SetString(PyExc_ValueError, "cudaMalloc error");
    }
    if (copyToDevice) CopyHostToDevice();
}

template <typename T> void NumCuda<T>::CopyDeviceToHost(){
    cudaError_t err = cudaSuccess;
	err = cudaMemcpy(hd, dd, nBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error copying from device memory: %s\n", cudaGetErrorString(err));
        PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
    }
}

template <typename T> void NumCuda<T>::CopyHostToDevice(){
    cudaError_t err = cudaSuccess;
	err = cudaMemcpy(dd, hd, nBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying to device memory: %s\n", cudaGetErrorString(err));
        PyErr_SetString(PyExc_ValueError, "cudaMemcpy error");
    }
}

template <typename T> PyArrayObject* NumCuda<T>::ExportNumpy(){
	T* copied_data = (T*)malloc(nBytes);
    memcpy(copied_data, hd, nBytes);
    return PyArray_SimpleNewFromData(ndims, dims, NpyTypeId, copied_data);
}