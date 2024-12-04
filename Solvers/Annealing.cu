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
int GPU_THREADS;

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
    //instead of throwing an error, we just return anyway and let the code die somewhere else
}

PyObject* Anneal(PyObject* self, PyObject* args, PyObject* kwargs) {

    //init from python inputs =======================================================================================================

    PyArrayObject* AnnealSchedule_np;
    PyObject* task;

    //these can be overridden by arguement
    PyArrayObject* InitialCondition_np = NULL;
    int nOptions = 1;
    int nActions = 1;
    const char *model_str = "PottsJit";
    const char *device_str = "CPU";
    const char *InclRepts = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int nReplicates = 1; 
    int nWorkers = 1;
    int nReports = -1;

    //parse input arguements, with python keyword capability:
    const char *kwlist[] = {"task", "PwlTemp", "IC", "nOptions", "nActions", "model", "device", "IncludedReports", "nReplicates", "nWorkers", "nReports", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Oiisssiii", const_cast<char **>(kwlist), //this cast could cause an error, but does not seem to
                                     &task, &AnnealSchedule_np, &InitialCondition_np, &nOptions, &nActions, &model_str, &device_str, &InclRepts, &nReplicates, &nWorkers, &nReports))
        return NULL;

    bool USE_GPU = false;
    if (!strcmp(device_str, "GPU") and GPU_AVAIL) USE_GPU = true;
    
    //process annealing schedule to get total niters:
    NumCuda<float> PwlSpecs(AnnealSchedule_np, 2, false, false);
    PieceWiseLinear PwlTemp(PwlSpecs);
    long niters = PwlSpecs(1, PwlSpecs.dims[1]-1);

    int nPartitions = PyLong_AsDouble(GetObjAttr(task, "nnodes"));
    // printf("Annealing problem with %i partitions\n", nPartitions);

    float e_th = PyFloat_AsDouble(GetObjAttr(task, "e_th"));

    
    bool PWLreports = false;
    if (nReports < 1){
        //an option to derive the report spacing from the PWL points, so as to allow more flexibility if desired
        PWLreports = true;
        nReports = PwlSpecs.dims[1]-1;
    }

    //weight and energy arrays that are used by ALL annealers.
    //"Best" and "Working" copies alternately provide "optimizer" and "sampling" views of the Boltzmann machine.
    NumCuda<int> BestStates(nReplicates, nPartitions);
    NumCuda<int> WrkStates(nReplicates, nPartitions);
    NumCuda<float> BestEnergies(nReplicates);
    NumCuda<float> WrkEnergies(nReplicates);
    
    //if an initial condition has been specified, initialize it into BestStates and WrkStates:
    if (InitialCondition_np != NULL) {
        NumCuda<int> InitialCondition(InitialCondition_np, -1, false, false); //only need this locally, to copy over data
        if (InitialCondition.ndims == 1 && InitialCondition.dims[0] == nPartitions){
            //one initial condition for all replicates
            for (int rep = 0; rep<nReplicates; rep++){
                for (int part = 0; part<nPartitions; part++){
                    BestStates(rep, part) = InitialCondition(part);
                    WrkStates(rep, part) = InitialCondition(part);
                }
            }
        }
        else if (InitialCondition.ndims == 2 && InitialCondition.dims[0] == nReplicates && InitialCondition.dims[1] == nPartitions){
            //unique initial condition for each replicate.
            for (int rep = 0; rep<nReplicates; rep++){
                for (int part = 0; part<nPartitions; part++){
                    BestStates(rep, part) = InitialCondition(rep, part);
                    WrkStates(rep, part) = InitialCondition(rep, part);
                }
            }
        }
        else printf("The supplied initial conditions does not have the right dimensions, it will be ignored\n");
        BestStates.CopyHostToDevice();
        WrkStates.CopyHostToDevice();

        //add "1" to all of the PWLSpecs here so that default iter=0 state initialization does not overwrite the specified initial condition

    }

    //a "communal" memory space for keeping track of which calculated options should be taken actions.
    //Is declared memory here, so that if the GPU version is used,
    //cooperatively working groups of threads can share information and memory does not have to be allocated within the kernel.
    //It is also co-opted to return the last "dwell time" from the annealing run,
    //which is needed to correctly scale sampling probabilities in some cases
    NumCuda<float> ActArb(nReplicates, nActions, 2);
    NumCuda<int> RealFlips(nReplicates);
    for (int rep = 0; rep<nReplicates; rep++) RealFlips(rep) = 0;
    RealFlips.CopyHostToDevice();

    //set up the global halt variable, used by the gpu to prematurely terminate a run across threads:
    NumCuda<int> GlobalHalt(1);
    *GlobalHalt.hd = -1;
    GlobalHalt.CopyHostToDevice();

    //set up output matrices.  These outputs only exist on the host side, and collect statistics as needed.
    //each one is optional (enabled by default), 
    //so that they can be excluded in case they become too burdensome from a memory perspective.
    NumCuda<float> reportMinEnergies;
    if (strchr(InclRepts, 'A') != NULL) reportMinEnergies = NumCuda<float>(nReports);
    NumCuda<float> reportAvgEnergies;
    if (strchr(InclRepts, 'B') != NULL) reportAvgEnergies = NumCuda<float>(nReports);
    NumCuda<float> reportAvgMinEnergies;
    if (strchr(InclRepts, 'C') != NULL) reportAvgMinEnergies = NumCuda<float>(nReports);
    NumCuda<float> reportAllEnergies;
    if (strchr(InclRepts, 'D') != NULL) reportAllEnergies = NumCuda<float>(nReports, nReplicates);
    NumCuda<int> reportAllStates;
    if (strchr(InclRepts, 'E') != NULL) reportAllStates = NumCuda<int>(nReports, nReplicates, nPartitions);
    NumCuda<int> reportAllMinStates;
    if (strchr(InclRepts, 'F') != NULL) reportAllMinStates = NumCuda<int>(nReports, nReplicates, nPartitions);
    NumCuda<int> reportMinStates;
    if (strchr(InclRepts, 'G') != NULL) reportMinStates = NumCuda<int>(nReports, nPartitions);
    NumCuda<float> reportDwellTimes; //for correcting sampling statistics
    if (strchr(InclRepts, 'H') != NULL) reportDwellTimes = NumCuda<float>(nReports, nReplicates);
    NumCuda<float> reportIter; //for convienience: iteration at which each report was measured
    if (strchr(InclRepts, 'I') != NULL) reportIter = NumCuda<float>(nReports); //should be type long to accomodate large numbers of iterations, but I think my NumCuda code is not fully set up for types other than float and int, so let's go with float
    NumCuda<float> reportTemp; //for convienience: annealing temperature when each report is recorded
    if (strchr(InclRepts, 'J') != NULL) reportTemp = NumCuda<float>(nReports);
    NumCuda<float> reportRealTimes; //wall-clock time of the execution
    if (strchr(InclRepts, 'K') != NULL) reportRealTimes = NumCuda<float>(nReports);
    NumCuda<int> reportRealFlips; //for convienience: iteration at which each report was measured
    if (strchr(InclRepts, 'L') != NULL) reportRealFlips = NumCuda<int>(nReports);

    //create an annealable object,
    //the flavor of which determines the "model" of how energy is determined from the state variables,
    //and what the permissible set of changes to the state are.
    Annealable *task_cpp;

    if (!strcmp(model_str, "PottsJit"))
        task_cpp = new PottsJitAnnealable(task, USE_GPU);

    else if (!strcmp(model_str, "PottsPrecompute"))
        task_cpp = new PottsPrecomputeAnnealable(task, nReplicates, USE_GPU);

    else if (!strcmp(model_str, "PottsPrecomputePE"))
        task_cpp = new PottsPrecomputePEAnnealable(task, nReplicates, USE_GPU);
        
    else if (!strcmp(model_str, "Tsp"))
        task_cpp = new TspAnnealable(task, USE_GPU, false);

    else if (!strcmp(model_str, "Tsp-extended"))
        task_cpp = new TspAnnealable(task, USE_GPU, true);

    else if (!strcmp(model_str, "Ising"))
        task_cpp = new IsingAnnealable(task, USE_GPU);

    else {
        printf("Error: model not recognized\n");
    }
        
    //loop in which the annealing function is called in segments.
    //report values are recorded at the end of each segment,
    //before annealing resumes with the same state that was left at the end of the previous run.
    long iters_per_report = niters/nReports;
    auto tStart = std::chrono::steady_clock::now();

    for (int r = 0; r < nReports; r++){
        long MinIter = PWLreports ? PwlSpecs(1, r) : r*iters_per_report;
        long MaxIter = PWLreports ? PwlSpecs(1, r+1) : (r+1)*iters_per_report;

        //in order to use a pre-defined IC (Initial Condition),
        //the first iteration is set to 1, preventing the task_cpp objects from doing random initialization,
        //which they do when they see iter=0.
        if (InitialCondition_np != NULL && MinIter == 0) MinIter = 1;


        //dispatch is a function pointer, set when the task is initialized.
        //The dispatch can point to either a CPU or GPU based function.
        //If GPU, the dispatch function handles reverse syncing of results from GPU memory to host memory.
        //Each function is customized to the task, i.e. is a template function compiled to use the specific task.
        //This organization works around two Cuda-related problems:
        //First, virtual function tables cannot be sent from host to GPU device,
        //so it works better to compile the annealing core specific to each task.
        //Second, by setting the function pointers in the class init,
        //the task-specific annealing cores are compiled alongside their class source files, and no cuda redistributable code is needed.
        task_cpp->dispatch(
            (void*)task_cpp, //passed as void, but each custom dispatch function casts task_cpp back to its specific object type.
            WrkStates,         BestStates,
            WrkEnergies,       BestEnergies,
            ActArb,
            RealFlips,
            PwlTemp,           nOptions, nActions, nWorkers,
            MinIter,           MaxIter,
            e_th,
            GlobalHalt);


        //collect performance/results:
        if (strchr(InclRepts, 'A') != NULL) reportMinEnergies(r) = BestEnergies.min();
        if (strchr(InclRepts, 'B') != NULL) reportAvgEnergies(r) = WrkEnergies.mean();
        if (strchr(InclRepts, 'C') != NULL) reportAvgMinEnergies(r) = BestEnergies.mean();

        if (strchr(InclRepts, 'D') != NULL) 
            for (int i = 0; i<nReplicates; i++)
                reportAllEnergies(r, i) = WrkEnergies(i);
        if (strchr(InclRepts, 'E') != NULL) 
            for (int i = 0; i<nReplicates; i++)
                for (int j = 0; j<nPartitions; j++)
                    reportAllStates(r, i, j) = WrkStates(i, j);
        if (strchr(InclRepts, 'F') != NULL) 
            for (int i = 0; i<nReplicates; i++)
                for (int j = 0; j<nPartitions; j++)    
                    reportAllMinStates(r, i, j) = BestStates(i,j);

        int best_indx = BestEnergies.ArgMin();
        if (strchr(InclRepts, 'G') != NULL) 
            for (int i = 0; i<nPartitions; i++)
                reportMinStates(r, i) = BestStates(best_indx, i);

        if (strchr(InclRepts, 'H') != NULL) 
            for (int i = 0; i<nReplicates; i++){
                //take average of the last set of arbitration times:
                reportDwellTimes(r, i) = ActArb(i, 0, 0);
            }

        if (strchr(InclRepts, 'I') != NULL) reportIter(r) = MaxIter;
        if (strchr(InclRepts, 'J') != NULL) reportTemp(r) = PwlTemp.interp(MaxIter);

        auto tNow = std::chrono::steady_clock::now();
        if (strchr(InclRepts, 'K') != NULL) reportRealTimes(r) = std::chrono::duration_cast<std::chrono::duration<float>>(tNow - tStart).count();

        if (strchr(InclRepts, 'L') != NULL) reportRealFlips(r) = RealFlips.mean();

    }

    //package results in a Python Dictionary:
    PyObject* reportDict = PyDict_New();
    if (strchr(InclRepts, 'A') != NULL) PyDict_SetItemString(reportDict, "MinEnergies", reportMinEnergies.ExportNumpy());
    if (strchr(InclRepts, 'B') != NULL) PyDict_SetItemString(reportDict, "AvgEnergies", reportAvgEnergies.ExportNumpy());
    if (strchr(InclRepts, 'C') != NULL) PyDict_SetItemString(reportDict, "AvgMinEnergies", reportAvgMinEnergies.ExportNumpy());
    if (strchr(InclRepts, 'D') != NULL) PyDict_SetItemString(reportDict, "AllEnergies", reportAllEnergies.ExportNumpy());
    if (strchr(InclRepts, 'E') != NULL) PyDict_SetItemString(reportDict, "AllStates", reportAllStates.ExportNumpy());
    if (strchr(InclRepts, 'F') != NULL) PyDict_SetItemString(reportDict, "AllMinStates", reportAllMinStates.ExportNumpy());
    if (strchr(InclRepts, 'G') != NULL) PyDict_SetItemString(reportDict, "MinStates", reportMinStates.ExportNumpy());
    if (strchr(InclRepts, 'H') != NULL) PyDict_SetItemString(reportDict, "DwellTimes", reportDwellTimes.ExportNumpy());
    if (strchr(InclRepts, 'I') != NULL) PyDict_SetItemString(reportDict, "Iter", reportIter.ExportNumpy());
    if (strchr(InclRepts, 'J') != NULL) PyDict_SetItemString(reportDict, "Temp", reportTemp.ExportNumpy());
    if (strchr(InclRepts, 'K') != NULL) PyDict_SetItemString(reportDict, "Time", reportRealTimes.ExportNumpy());
    if (strchr(InclRepts, 'L') != NULL) PyDict_SetItemString(reportDict, "RealFlips", reportRealFlips.ExportNumpy());

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
    User can choose the annealing model, and specify what outputs they want back."}, //this last one is a doc string.
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
        GPU_THREADS = 0;
    } else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, /*deviceCount=*/ 0);
        float mem_gb = static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024);
        printf("GPU %s is available, with %i streaming multiprocessors and %.2f GB memory \n", deviceProp.name, deviceProp.multiProcessorCount, mem_gb);
        GPU_AVAIL = true;
        GPU_THREADS = deviceProp.multiProcessorCount*deviceProp.maxThreadsPerMultiProcessor;
    }

    return PyModule_Create(&AnnealingModule);
}