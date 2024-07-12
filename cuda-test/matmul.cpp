#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdbool.h>
#include <random>

// Forward function declaration.
static PyObject *matmul_cpu(PyObject *self, PyObject *args);
static PyObject *matmul_gpu(PyObject *self, PyObject *args);

//method list. One entry for each method/function
static PyMethodDef methods[] = {
    {"matmul_cpu", matmul_cpu, METH_VARARGS, "Doc string."},
    {"matmul_gpu", matmul_gpu, METH_VARARGS, "Doc string."},
    {NULL, NULL, 0, NULL} //this line of junk is added so that methods[] is thought of as an array and not just a pointer? huh?
};



static struct PyModuleDef matmulModule = {
    PyModuleDef_HEAD_INIT,
    "matmul",
    "some custom matrix multiplication",
    -1,
    methods
};


PyMODINIT_FUNC PyInit_matmul(void){
    import_array(); //Grrr! This imports some things, such as PyArray_SimpleNewFromData.  Without this line, code will still compile, but will seg fault
    return PyModule_Create(&matmulModule);
}



static PyObject* matmul_cpu(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;

    if (!PyArg_ParseTuple(args, "OO", &array1, &array2))
        return NULL;

    // PyObject *edges_arg=NULL, *coloring_arg=NULL;
    // PyObject *array1=NULL, *array2=NULL;

    // // Parse arguments. 
    // if (!PyArg_ParseTuple(args, "OO",
    //                       &array1, 
    //                       &array2)) {
    //   return NULL;
    // }


    if (array1 -> nd != 2 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];

    //re-frame arrays for easier C++ access:
    double * states = (double *)array2->data;
    double * weights = (double *)array1->data;



    //array1->strides[0]
    // double f1 = weights[1*n1 + 1];
    // double f2 = weights[2*n1 + 1];

    //uh.  Let's just see if we can access the data correctly:
    // printf("Weight matrix values are: %.2f, %.2f\n", f1, f2);

    // if (n1 != n2) {
    //     PyErr_SetString(PyExc_ValueError, "arrays must have the same length");
    //     return NULL;
    // }

    double * output = (double *) malloc(sizeof(double) * n2);

    //slower way - wrong memory order:
    // for (int i = 0; i < n2; i++){
    //     double ps = 0; //partial sum
    //     for (int j = 0; j < n1; j++)
    //       ps = ps + weights[j*n1+i]*states[i];
    //     output[i] = ps;
    // }

    for (int i = 0; i < n2; i++) //initialize to zero
      output[i] = 0;

    for (int j = 0; j < n1; j++)
        for (int i = 0; i < n2; i++)
          output[i] = output[i] + weights[j*n1+i]*states[i];

    npy_intp const dims [1] = {n1};
    PyObject* array3 = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, output);
    // Py_RETURN_NONE;
    return array3;
}

double * myVectorAdd(double * h_A, double * h_B, int numElements);

static PyObject* matmul_gpu(PyObject* self, PyObject* args) {
    PyArrayObject* array1, * array2;

    if (!PyArg_ParseTuple(args, "OO", &array1, &array2))
        return NULL;

    if (array1 -> nd != 2 || array2 -> nd != 1 || array1->descr->type_num != PyArray_DOUBLE || array2->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
        return NULL;
    }

    int n1 = array1->dimensions[0];
    int n2 = array2->dimensions[0];
    printf("n1: %i\n", n1);

    //re-frame arrays for easier C++ access:
    double * states = (double *)array2->data;
    double * weights = (double *)array1->data;

    double * output = myVectorAdd(weights, states, n1);

    npy_intp const dims [1] = {n1};
    PyObject* array3 = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, output);
    // Py_RETURN_NONE;
    return array3;
}




/* note to self about c arrays:

array2 //is a pointer to a PyArrayObject
array2->data //is a pointer to the start of memory containing the array data, but without knowing anything about the data type
(double *)array2->data //is a pointer to the start of data memory, now with the understanding that it is an array of doubles
(double *) array2 -> data + i //is now a pointer to the memory address of data i
*((double *) array2 -> data + i) //dereferences the pointer, i.e. actually gets the data stored at that particular pointer location

*/