#define PY_SSIZE_T_CLEAN
#include <Python.h>
// #include <numpy/arrayobject.h>

//no real code in this file, only definitions for all the various C++ and Cuda functions designed for python interfacing
//what, does that mean this is really a header file? duuhhhhh

// Forward function declaration.
//these need to be static here, but not declared static in their actual implimentation
static PyObject *PottsGpuFwSolve(PyObject *self, PyObject *args);
static PyObject *PottsGpuKwSolve(PyObject *self, PyObject *args);
static PyObject *PottsCpuStSolve(PyObject *self, PyObject *args);

// static PyObject *TestFmt(PyObject *self, PyObject *args);

static PyObject *ColoringNonRedundantSearch(PyObject *self, PyObject *args);
static PyObject *ColoringStreamlinedPotts(PyObject *self, PyObject *args);


//method list. One entry for each method/function
static PyMethodDef methods[] = {
    {"PottsGpuFwSolve",             PottsGpuFwSolve,            METH_VARARGS, "Doc string."},
    {"PottsGpuKwSolve",             PottsGpuKwSolve,            METH_VARARGS, "Doc string."},
    {"PottsCpuStSolve",             PottsCpuStSolve,            METH_VARARGS, "Doc string."},
    {"ColoringNonRedundantSearch",  ColoringNonRedundantSearch, METH_VARARGS, "Doc string."},
    {"ColoringStreamlinedPotts",    ColoringStreamlinedPotts,   METH_VARARGS, "Doc"},
    // {"TestFmt",                     TestFmt,                    METH_VARARGS | METH_KEYWORDS, "a test"},
    {NULL, NULL, 0, NULL} //this line of junk is added so that methods[] is thought of as an array and not just a pointer? huh?
};


static struct PyModuleDef AllSolversModule = {
    PyModuleDef_HEAD_INIT,
    "AllSolvers",
    "Solve combinatorial optimization problems",
    -1,
    methods
};


PyMODINIT_FUNC PyInit_AllSolvers(void){
    // import_array(); //Grrr! This imports some things, such as PyArray_SimpleNewFromData.  Without this line, code will still compile, but will seg fault
    return PyModule_Create(&AllSolversModule);
}