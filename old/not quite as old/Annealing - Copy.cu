//python includes:
#define PY_SSIZE_T_CLEAN //why? not sure.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <numpy/ndarrayobject.h>

//regular includes:
#include <stdio.h>

//CUDA includes:
#include <cuda_runtime.h>
#include <curand_kernel.h>

//my includes:
#include "NumCuda.hpp"
#include "PieceWiseLinear.cpp"
#include "PottsFullWeights.cu"
#include "PottsKernelWeights.cu"
#include "PottsEfirstCooperative.cu"


PyObject* GetObjAttr(PyObject* obj, const char* name){

    if(PyDict_Check(obj)){
        //dictionary retrieval:
        if (!PyDict_Contains(obj, PyUnicode_FromString(name))) printf("Error, object does not have attribute %s\n", name);
        return PyDict_GetItemString(obj, name);
    }
    else{
        if (!PyObject_HasAttr(obj, PyUnicode_FromString(name))) printf("Error, object does not have attribute %s\n", name);
        return PyObject_GetAttr(obj, PyUnicode_FromString(name));
    }
    //instead of throwing an error, we just return anyway and let the code die somewhere else:    
}



// PyObject* ArgsToFile(PyObject* self, PyObject* args, PyObject* kwargs) {
//     //saves self, args, and kwargs to pickle files.
//     //later, they can be loaded entirely from within C++,
//     //so that the python-generated inputs can be used during debug runs with gdb and gdb-cuda!

//     //instead of writing file acess and pickle calls in C,
//     //let's try doing that heavy work in Python and then calling those functions from C. (maybe?)
//     static PyObject *module = PyImport_ImportModuleNoBlock("ArgPickler"); 

//     pkl_self = 

// }


PyObject* Anneal(PyObject* self, PyObject* args, PyObject* kwargs) {

    //init from python inputs =======================================================================================================
    import_array(); //numpy initialization function, otherwise may get seg faults
    // Py_Initialize();//otherwise the dictionary at the end will not work
    PyArrayObject* AnnealSchedule_np;
    PyObject* task;

    //these can be overridden by arguement
    const char *method_str = "Cpu";
    int nReplicates = 1; 
    int nReports = 1;

    // printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));

    //parse input arguements, with python keyword capability:
    const char *kwlist[] = {"task", "PwlTemp", "method", "nReplicates", "nReports", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|sii", const_cast<char **>(kwlist), //this cast could cause an error, but does not seem to
                                     &task, &AnnealSchedule_np, &method_str, &nReplicates, &nReports))
        return NULL;

    //GPU parallelism parameters
    int threadsPerBlock = 32;
    int blocksPerGrid = (nReplicates + threadsPerBlock - 1) / threadsPerBlock;


    //process annealing schedule to get total niters:
    NumCuda<float> PwlSpecs(AnnealSchedule_np, 2, false, false);
    PieceWiseLinear PwlTemp(PwlSpecs);
    int niters = PwlSpecs(1, PwlSpecs.dims[0]-1);

    int nPartitions = PyLong_AsDouble(GetObjAttr(task, "nnodes"));
    // printf("Annealing problem with %i partitions\n", nPartitions);

    float e_th = PyFloat_AsDouble(GetObjAttr(task, "e_th"));

    //problem specifications.
    //Depending on the Annealer, different spec formats are required,
    //so these are empty unless the specific annealer is being used.

    //annealing with block-matrix weight compression:
    NumCuda<float> kernels;
    NumCuda<int> kmap;
    NumCuda<int> qSizes;
    if (!strcmp(method_str, "GpuKernel") || !strcmp(method_str, "CpuKernel")){
        kernels = NumCuda<float>((PyArrayObject*)GetObjAttr(task, "kernels"), 3, false, true);
        kmap    = NumCuda<int>  ((PyArrayObject*)GetObjAttr(task, "kmap"), 2, false, true);
        qSizes  = NumCuda<int>  ((PyArrayObject*)GetObjAttr(task, "qSizes"), 1, false, true);
    }
    
    //annealing with a full weight matrix:
    NumCuda<float> weights;
    NumCuda<int> Partitions;
    int nNHPPs = 1;
    if (!strcmp(method_str, "Gpu") || !strcmp(method_str, "Cpu") || !strcmp(method_str, "GpuEfirstCooperative")){
        weights = NumCuda<float>((PyArrayObject*)GetObjAttr(task, "weights"), 2, false, true);
        Partitions = NumCuda<int>((PyArrayObject*)GetObjAttr(task, "Partitions"), 1, false, true);
        nNHPPs = weights.dims[0];
        // printf("nNHPPs: %i\n", nNHPPs);
    }
    // Py_RETURN_NONE;
    // printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
    // 
    //special-format; use depends on how the task-specific solver is defined.
    //float and int options, a particular task solver may only use one of these.
    NumCuda<float> TaskFormF;
    NumCuda<int> TaskFormI;
    if (!strcmp(method_str, "TaskSpecific")){
        TaskFormF = NumCuda<float>((PyArrayObject*)GetObjAttr(task, "TaskFormF"), -1, false, true);
        TaskFormI = NumCuda<int>  ((PyArrayObject*)GetObjAttr(task, "TaskFormI"), -1, false, true);
    }

    //weight and energy arrays that are used by ALL annealers.
    //"Best" and "Working" copies alternately provide "optimizer" and "sampling" views of the Boltzmann machine.
    int stateDims[2] = {nReplicates, nPartitions};
    NumCuda<int> BestStates(2, stateDims);
    NumCuda<int> WrkStates(2,stateDims);
    NumCuda<float> BestEnergies(1, &nReplicates);
    NumCuda<float> WrkEnergies(1, &nReplicates);

    //set up the global halt variable, used by the gpu to prematurely terminate a run across threads:
    int one = 1;
    NumCuda<int> GlobalHalt(1, &one);
    *GlobalHalt.hd = -1;
    GlobalHalt.CopyHostToDevice();

    //some scratch space memory, for various purposes
    int NhppPotentialsDims[2] = {nReplicates, nNHPPs};
    NumCuda<float> NhppPotentials(2, NhppPotentialsDims);
    int PtpDims[3] = {nReplicates, threadsPerBlock, 2};
    NumCuda<float> PerThreadProposals(3, PtpDims);

    //set up output matrices.  These outputs only exist on the host side, and collect statistics as needed.
    NumCuda<float> reportMinEnergies(1, &nReports);
    NumCuda<float> reportAvgEnergies(1, &nReports);
    NumCuda<float> reportAvgMinEnergies(1, &nReports);

    int reportAllEnergiesDims[2] = {nReports, nReplicates};
    NumCuda<float> reportAllEnergies(2, reportAllEnergiesDims);
    int reportMinStatesDims[2] = {nReports, nPartitions};
    NumCuda<int> reportMinStates(2, reportMinStatesDims);
    int reportAllMinStatesDims[3] = {nReports, nReplicates, nPartitions};
    NumCuda<int> reportAllMinStates(3, reportAllMinStatesDims);
    int reportAllStatesDims[3] = {nReports, nReplicates, nPartitions};
    NumCuda<int> reportAllStates(3, reportAllStatesDims);

    //these last two are just for convienience, for plotting results back in python
    NumCuda<int> reportIter(1, &nReports); 
    NumCuda<float> reportTemp(1, &nReports);


    // printf("Partitions(0)=%i\n",Partitions(0));
    //loop in which the annealing function is called in segments.
    //report values are recorded at the end of each segment,
    //before annealing resumes with the same state that was left at the end of the previous run.
    int iters_per_report = niters/nReports;
    for (int r = 0; r < nReports; r++){
        int MinIter = r*iters_per_report;
        int MaxIter = (r+1)*iters_per_report;


        // annealing method run: one of these will be used
        if (!strcmp(method_str, "Gpu"))
            PottsFullWeightsDevice<<<blocksPerGrid, threadsPerBlock>>>(
                weights,        Partitions,
                WrkStates,      BestStates,
                WrkEnergies,    BestEnergies,
                MinIter,        MaxIter,
                PwlTemp,        e_th,               GlobalHalt.dd);

        else if (!strcmp(method_str, "GpuKernel"))
            KernelWeightsGpu<<<blocksPerGrid, threadsPerBlock>>>(
                kernels,        kmap,           qSizes,
                WrkStates,      BestStates,
                WrkEnergies,    BestEnergies,
                MinIter,        MaxIter,
                PwlTemp,        e_th,               GlobalHalt.dd);

        else if (!strcmp(method_str, "GpuEfirstCooperative"))
            EfirstCooperative<<<nReplicates, threadsPerBlock>>>(
                weights,        Partitions,
                WrkStates,      BestStates,
                WrkEnergies,    BestEnergies,
                NhppPotentials, PerThreadProposals,
                MinIter,        MaxIter,
                PwlTemp,        e_th,               GlobalHalt.dd);

        else if (!strcmp(method_str, "Cpu"))
            PottsFullWeightsHost(
                weights,        Partitions,
                WrkStates,      BestStates,
                WrkEnergies,    BestEnergies,
                MinIter,        MaxIter,
                PwlTemp,        e_th,               GlobalHalt.hd);

        else if (!strcmp(method_str, "CpuKernel"))
            KernelWeights(
                kernels,        kmap,           qSizes,
                WrkStates,      BestStates,
                WrkEnergies,    BestEnergies,
                MinIter,        MaxIter,
                PwlTemp,        e_th,               GlobalHalt.hd);


        //cleanup after a GPU run:
        if (!strcmp(method_str, "Gpu") || !strcmp(method_str, "GpuKernel") || !strcmp(method_str, "GpuEfirstCooperative")){
            cudaDeviceSynchronize(); //wait for GPU processing to finish
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
                PyErr_SetString(PyExc_ValueError, "CUDA kernel execution error");
                return NULL;
            }
            BestStates.CopyDeviceToHost();
            WrkStates.CopyDeviceToHost();
            BestEnergies.CopyDeviceToHost();
            WrkEnergies.CopyDeviceToHost();
        }

        // printf("We're probably segfaulting while copying results...\n");
        //collect performance/results:
        reportMinEnergies(r) = BestEnergies.min();
        reportAvgEnergies(r) = WrkEnergies.mean();
        reportAvgMinEnergies(r) = BestEnergies.mean();

        for (int i = 0; i<nReplicates; i++)
            reportAllEnergies(r, i) = WrkEnergies(i);
        for (int i = 0; i<nReplicates; i++)
            for (int j = 0; j<nPartitions; j++)
                reportAllStates(r, i, j) = WrkStates(i, j);
        for (int i = 0; i<nReplicates; i++)
            for (int j = 0; j<nPartitions; j++)    
                reportAllMinStates(r, i, j) = BestStates(i,j);

        int best_indx = BestEnergies.ArgMin();
        for (int i = 0; i<nPartitions; i++)
            reportMinStates(r, i) = BestStates(best_indx, i);

        reportIter(r) = MaxIter;
        reportTemp(r) = PwlTemp.interp(MaxIter);

    }

    //package results in a Python Dictionary:
    PyObject* reportDict = PyDict_New();
    PyDict_SetItemString(reportDict, "MinEnergies", reportMinEnergies.ExportNumpy());
    PyDict_SetItemString(reportDict, "AvgEnergies", reportAvgEnergies.ExportNumpy());
    PyDict_SetItemString(reportDict, "AvgMinEnergies", reportAvgMinEnergies.ExportNumpy());
    PyDict_SetItemString(reportDict, "AllEnergies", reportAllEnergies.ExportNumpy());
    PyDict_SetItemString(reportDict, "AllStates", reportAllStates.ExportNumpy());
    PyDict_SetItemString(reportDict, "AllMinStates", reportAllMinStates.ExportNumpy());
    PyDict_SetItemString(reportDict, "MinStates", reportMinStates.ExportNumpy());
    PyDict_SetItemString(reportDict, "Iter", reportIter.ExportNumpy());
    PyDict_SetItemString(reportDict, "Temp", reportTemp.ExportNumpy());

    return reportDict;
    Py_RETURN_NONE;
}


//for C++ only debug, reads python-generated inputs from a file and sends them to main function.
//should allow debugging with gdb and gdb-cuda
void printobj(PyObject * obj){
    PyObject* ObjStr = PyObject_Str(obj);
    const char* s = PyUnicode_AsUTF8(ObjStr);
    printf("%s\n",s);
}

int main(){
    // import_array(); //numpy initialization function, otherwise may get seg faults
    Py_Initialize(); //initialized python interpreter?
    static PyObject *pickle = PyImport_ImportModuleNoBlock("pickle");
    printobj(pickle);

    static PyObject *io = PyImport_ImportModuleNoBlock("io");
    printobj(io);

    PyObject *f = PyObject_CallMethod(io, "open", "OO", PyUnicode_FromString("ArgsFile.pkl"), PyUnicode_FromString("rb"));
    printobj(f);
    // PyObject *unpickler = PyObject_CallMethod(pickle, "Unpickler", "O", ); //third arg is list of the function's arguements and their types.
    // printobj(unpickler);
    PyObject *d = PyObject_CallMethod(pickle, "load", "O", f);
    // printobj(args_tuple);
    PyObject *args = GetObjAttr(d, "args");
    PyObject *kwargs = GetObjAttr(d, "kwargs");
    PyObject* self = PyDict_New(); //use an empty dict as the self arg, since I don't use it in the code? Hmm...
    PyObject* ret = Anneal(self, args, kwargs);

    printf("Well, I didn't crash\n");
    return 0;
}






// =================================================================================Python module initialization code - mostly boilerplate

static PyMethodDef methods[] = {
    {"Anneal", (PyCFunction) Anneal, METH_VARARGS | METH_KEYWORDS, 
    "Runs simulated annealing steps on a given problem. \
    User can choose the annealing backend, and specify what outputs they want back."}, //this last one is a doc string.
    {NULL, NULL, 0, NULL} //this line of junk is added so that methods[] is thought of as an array and not just a pointer? huh?
};


static struct PyModuleDef AnnealingModule = {
    PyModuleDef_HEAD_INIT,
    "Annealing",
    "Solve combinatorial optimization problems",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_Annealing(void){
    // import_array(); //Grrr! This imports some things, such as PyArray_SimpleNewFromData.  Without this line, code will still compile, but will seg fault
    return PyModule_Create(&AnnealingModule);
}