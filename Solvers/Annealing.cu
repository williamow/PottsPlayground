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

//my includes
#include "NumCuda.hpp"
#include "PieceWiseLinear.cpp"
#include "Annealables.h"

//global switch, set at init time, to tell if GPU can be used or not:
bool GPU_AVAIL;

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

PyObject* Anneal(PyObject* self, PyObject* args, PyObject* kwargs) {

    //init from python inputs =======================================================================================================
    
    // Py_Initialize();//otherwise the dictionary at the end will not work
    PyArrayObject* AnnealSchedule_np;
    PyObject* task;

    //these can be overridden by arguement
    int OptsPerThrd = 1;
    bool TakeAllOptions = true;
    const char *backend_str = "PottsJit";
    const char *substrate_str = "CPU";
    int nReplicates = 1; 
    int nWorkers = 1;
    int nReports = 1;

    // printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));

    //parse input arguements, with python keyword capability:
    const char *kwlist[] = {"task", "PwlTemp", "OptsPerThrd", "TakeAllOptions", "backend", "substrate", "nReplicates", "nWorkers", "nReports", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ibssiii", const_cast<char **>(kwlist), //this cast could cause an error, but does not seem to
                                     &task, &AnnealSchedule_np, &OptsPerThrd, &TakeAllOptions, &backend_str, &substrate_str, &nReplicates, &nWorkers, &nReports))
        return NULL;


    bool USE_GPU = false;
    if (!strcmp(substrate_str, "GPU") and GPU_AVAIL) USE_GPU = true;


    //process annealing schedule to get total niters:
    NumCuda<float> PwlSpecs(AnnealSchedule_np, 2, false, false);
    PieceWiseLinear PwlTemp(PwlSpecs);
    int niters = PwlSpecs(1, PwlSpecs.dims[1]-1);

    int nPartitions = PyLong_AsDouble(GetObjAttr(task, "nnodes"));
    // printf("Annealing problem with %i partitions\n", nPartitions);

    float e_th = PyFloat_AsDouble(GetObjAttr(task, "e_th"));

    
    

    //weight and energy arrays that are used by ALL annealers.
    //"Best" and "Working" copies alternately provide "optimizer" and "sampling" views of the Boltzmann machine.
    NumCuda<int> BestStates(nReplicates, nPartitions);
    NumCuda<int> WrkStates(nReplicates, nPartitions);
    NumCuda<float> BestEnergies(nReplicates);
    NumCuda<float> WrkEnergies(nReplicates);

    //a "communal" space for cooperatively working groups of threads to share information
    NumCuda<float> commspace(nReplicates, nWorkers, 10);

    //set up the global halt variable, used by the gpu to prematurely terminate a run across threads:
    NumCuda<int> GlobalHalt(1);
    *GlobalHalt.hd = -1;
    GlobalHalt.CopyHostToDevice();

    //set up output matrices.  These outputs only exist on the host side, and collect statistics as needed.
    NumCuda<float> reportMinEnergies(nReports);
    NumCuda<float> reportAvgEnergies(nReports);
    NumCuda<float> reportAvgMinEnergies(nReports);

    NumCuda<float> reportAllEnergies(nReports, nReplicates);
    NumCuda<int> reportMinStates(nReports, nPartitions);
    NumCuda<int> reportAllMinStates(nReports, nReplicates, nPartitions);
    NumCuda<int> reportAllStates(nReports, nReplicates, nPartitions);

    //these last two are just for convienience, for plotting results back in python
    NumCuda<int> reportIter(nReports); 
    NumCuda<float> reportTemp(nReports);

    //create an annealable object,
    //the flavor of which determines the "model" of how energy is determined from the state variables,
    //and what the permissible set of changes to the state are.
    Annealable *task_cpp;

    if (!strcmp(backend_str, "PottsJit"))
        task_cpp = new PottsJitAnnealable(task, USE_GPU);
        
    else if (!strcmp(backend_str, "Tsp"))
        task_cpp = new TspAnnealable(task, USE_GPU);
        

    //loop in which the annealing function is called in segments.
    //report values are recorded at the end of each segment,
    //before annealing resumes with the same state that was left at the end of the previous run.
    int iters_per_report = niters/nReports;
    for (int r = 0; r < nReports; r++){
        int MinIter = r*iters_per_report;
        int MaxIter = (r+1)*iters_per_report;


        //dispatch is a function pointer, set when the task is initialized.
        //The dispatch can point to either a CPU or GPU based function.
        //If GPU, the dispatch function handles reverse syncing of results from GPU memory to host memory.
        //Each function is costomized to the task, i.e. is a template function compiled to use the specific task.
        //This organization works around two Cuda-related problems:
        //First, virtual function tables cannot be sent from host to GPU device,
        //so it works better to compile the annealing core specific to each task.
        //Second, by setting the function pointers in the class init,
        //the task-specific annealing cores are compiled alongside their class source files, and no cuda redistributable code is needed.
        task_cpp->dispatch(
            (void*)task_cpp, //passed as void, but each custom dispatch function casts task_cpp back to its specific object type.
            WrkStates,         BestStates,
            WrkEnergies,       BestEnergies,
            commspace,
            PwlTemp,           OptsPerThrd, TakeAllOptions,
            MinIter,           MaxIter,
            e_th,
            GlobalHalt);


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

//initialization function, called when python module is imported
PyMODINIT_FUNC PyInit_Annealing(void){
    import_array(); //Grrr! This imports some things, such as PyArray_SimpleNewFromData.  Without this line, code will still compile, but will seg fault
    
    //detect and initialize GPU, if there is one:
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No GPUs detected, defaulting to CPU-only operation.\n");
        GPU_AVAIL = false;
    } else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, /*deviceCount=*/ 0);
        float mem_gb = static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024);
        printf("GPU %s is available, with %i streaming multiprocessors and %.2f GB memory \n", deviceProp.name, deviceProp.multiProcessorCount, mem_gb);
        GPU_AVAIL = true;
    }

    return PyModule_Create(&AnnealingModule);
}