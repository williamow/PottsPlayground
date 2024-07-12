#define PY_SSIZE_T_CLEAN //why? not sure.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <stdio.h>
#include <numpy/ndarrayobject.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "NumCuda.hpp"


/*
Okay, so: you can easily write the energy and state sampling code that you want.
The challenge is, how to do it without increasing compexity, and without creating parallel instances of the same underlying algorithm?
Having several different C interfaces is Okay.  Would be nice if we had one kernel, and several C interfaces.
Actually, the Python to C and C to Cuda interfaces take a lot of space, so maybe it would be good to not replicate those either.  Hmm uuughhh
Let's work backwards from what top-level API should look like:

    Potts_solve(kernels, kmap, qSizes, etc)
    Potts_sample()
    Potts_energies()

Well really you could send all data back all the time, but that would slow down the software significantly.  Large overhead.  nope
Could do serial kernel invocations. As long as the niters within each invocation is long enough, the overhead of repeated kernel launches would be negligible?
And if not negligible, that means we are trying to collect many, many samples, which would incur high overhead any way we do it.

Maybe, your project for today:
-create a class called NumCuda, which can initialize from a Numpy array object and cleanly copy the array over to CUDA and Back,
i.e. internally keeps track of the different array realities.

Possible API:

Ah. but how would we make this for all the different array data types that are possible? Maybe....

NumCuda.New(ndims, *dims, type) #create a new data array, allocating memory on both host and device
NumCuda.NewFromPy(ndims,type) #can integrate data type and dimension checking in this function. Maybe have it copy data, rather than referencing the Numpy data

NumCuda.syncHtoD()
NumCuda.syncDtoH()
NumCuda.dd[] #device data access
NumCuda.hd[] #host data access
NumCuda.sz() #total number of 
NumCuda.shape[]

NumCuda.Numpy() #returns a copy of the device NumCuda data in a python numpy object format
NumCuda.del() #clears all the allocated host and device memory



cuda kernel for sampling a single replicate of a Potts model.

weights: nNHPPs x nNHPPs
	matrix of weights in the boltzmann machine.  Symmetric, row/column ordering can be chosen to maximize performance.
partitions: nNHPPs
    indicates which Potts group each NHPP belongs to; maps NHPP index to addresses in the state vectors.
    Each integer value is therefore in the range [0, nPartitions)
states: nReplicates x nPartitions
	Each state value indicates the active state of a Potts node, represented by the integer index of the NHPP,
    so that the values in the state vectors directly index to which values in the weight matrix are 'active' for the given state.
    Two sets of state memory are provided - one for keeping track of the best-so-far state,
    and another for holding the working, evolving state that the algorithm acts upon. 
*/
__global__ void potts_AllInOne1(
    NumCuda<float> kernels,
    NumCuda<int> kmap,
    NumCuda<int> qSizes,
    int *working_states,
    int *best_states,
    float *working_energies,
    float *best_energies,
    int MinIter,
    int MaxIter,
    int nReplicates,
    float Tmax, //settings to specify a sawtooth thermal annealing profile
    float Tmin,
    int Tperiod,
    float e_th, //energy threshold to trigger premature termination
    volatile int *global_halt //bit that tells all processes to exit, set by a single process upon finding a 'solution'.
    //super important that this is declared volatile, so that it is always retreived from global memory, 
    )
{
    
    //index for which replicate this thread is computing; only matters for writing the final best state back to global memory
    int replicate_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (replicate_index + 1 > nReplicates) return;

    int nPartitions = qSizes.dims[0];

    //create local references to the state vectors used in this particular thread:
    int *MiWrk = &working_states[replicate_index*nPartitions];
    int *MiBest = &best_states[replicate_index*nPartitions];


    //rng initialization ==========================================================================

    //the seeds are all the same, with different "offsets" from the seed state, depending on the kernel's replicate number.

    //seed value for random number generator that decides whether a state transition occurs or not:
    int flip_seed = blockDim.x * blockIdx.x + threadIdx.x;
    curandStateMRG32k3a_t flipRngState;
    curand_init(flip_seed, MinIter, 0, &flipRngState);

    //seed value for random number generator that decides which state transitions to propose.
    //by sharing this seed between all threads in a block, memory accesses should be reduced, since the same weights will be requested each time.
    // int trial_seed = 10000 + blockDim.x * blockIdx.x;
    // curandStateMRG32k3a_t trialRngState;
    // curand_init(0, trial_seed, 0, &trialRngState);

    float Tmod = (Tmax-Tmin)/Tperiod;

    //total energy tracker initialization ===============================================

    if (MinIter == 0){
        //initialize states, only if we are starting the first round of calculation
        for (int partition = 0; partition < nPartitions; partition++)
            MiWrk[partition] = curand(&flipRngState)%qSizes(partition); //random state initialization
    }


    //calculate starting total energy:
    float current_e = 0;
    for (int i = 0; i < nPartitions; i++){
        for (int j = 0; j < nPartitions; j++){
            current_e += kernels(kmap(i,j), MiWrk[i], MiWrk[j]);//[   qMax * qMax *kmap(i, j)  +  MiWrk[i]*qMax  +  MiWrk[j]  ];
            //weights contribute to the active energy when both sides of the weights are selected.
        }
    }
    current_e = current_e / 2;

    // printf("intial e: %.2f\n", current_e);
    // return;
    float lowest_e = current_e;
    for (int i = 0; i < nPartitions; i++){
        MiBest[i] = MiWrk[i];
    }
    if (lowest_e < e_th){
        *global_halt = replicate_index;
        __threadfence_system(); //make the global halt value visible to all other threads
    }
            

    //main loop ==================================================================================
    for (int iter = MinIter; iter < MaxIter; iter++){

        if (*global_halt > 0)
            break; //if one of the threads has found a solution, all threads exit here

        int wrk_part = curand(&flipRngState)%nPartitions;
        int Mi_proposed = curand(&flipRngState)%qSizes(wrk_part);
        int Mi_current = MiWrk[wrk_part];
        if (Mi_proposed == Mi_current) continue;

        float dE = 0;
        //compute how much the energy would change.
        //assumes no weights between Mi_proposed and Mi_current, otherwise this calculation would be incorrect.
        for (int i = 0; i < nPartitions; i++){
            dE = dE +kernels(kmap(wrk_part, i), Mi_proposed, MiWrk[i]) //[   qMax * qMax *kmap(wrk_part, i)  +  Mi_proposed*qMax +  MiWrk[i]  ]
                    -kernels(kmap(wrk_part, i), Mi_current, MiWrk[i]); //[   qMax * qMax *kmap(wrk_part, i)  +  Mi_current*qMax  +  MiWrk[i]  ];
        }

        float T = Tmod*float(iter%Tperiod) + Tmin;
        float flip_prob = exp(-dE/T);

        if (flip_prob >= curand_uniform(&flipRngState)){
            //update the state:
            MiWrk[wrk_part] = Mi_proposed;
            //update the energy:
            current_e += dE;
            //and possibly update the lowest energy value:
            if (current_e < lowest_e){
                lowest_e = current_e;
                for (int i = 0; i < nPartitions; i++){
                    MiBest[i] = MiWrk[i];
                }
                if (lowest_e < e_th){
                    *global_halt = 1;
                    // __threadfence_system(); //make the global halt value visible to all other threads
                }
            }
        }
    }
    working_energies[replicate_index] = current_e;
    best_energies[replicate_index] = lowest_e;
    return;
}

PyObject* TestFmt(PyObject* self, PyObject* args, PyObject* kwargs) {

    PyObject *d1, *d2;

    if (!PyArg_ParseTuple(args, "OO", &d1, &d2))
        return NULL;

    if (PyDict_Check(d1)) printf("D1 is a dict!\n"); else printf("D1 is not a dict!\n");
    if (PyDict_Check(d2)) printf("D2 is a dict!\n"); else printf("D2 is not a dict!\n");

    Py_RETURN_NONE;

    //some functional code that I worked on elsewhere and am dumping here because I don't quite want to delete it or look at it:
    //create some simple python objects, so we can do matching and comparison in python space:
    PyObject* method_block = PyUnicode_FromString("block");
    PyObject* method = PyDict_GetItem(AnnealerOptions, PyUnicode_FromString("method"));

    if (PyObject_RichCompareBool(method, method_block, Py_EQ)){
        printf("Match found!\n");
    }

    if (strcmp(method_str, "head")){
        printf("Method match found!\n");
    }
    
}


PyObject* PottsGpuKwSolve(PyObject* self, PyObject* args) {

    //init from python inputs =======================================================================================================
    import_array(); //numpy initialization function, otherwise may get seg faults
    PyArrayObject* kernels_np, *kmap_np, *qSizes_np, *schedule_np;

    //schedule is... a piecewise-linear description of the temperature annealing schedule.
    int nReplicates, niters;
    float Tmin, Tmax, e_th;
    int Tperiod, nBreaks;

    if (!PyArg_ParseTuple(args, "OOOiiiffif", &kernels_np, &kmap_np, &qSizes_np, &schedule_np, &nReplicates, &niters, &nBreaks, &e_th))
        return NULL;

    //transfer and convert numpy arrays to arrays in the GPU:
    NumCuda<float> kernels(kernels_np, 3, false, true);
    NumCuda<int> kmap(kmap_np, 2, false, true);
    NumCuda<int> qSizes(qSizes_np, 1, false, true);
    NumCuda<float> schedule(schedule_np, 2, false, true);
    
    int nPartitions = PyArray_SHAPE(kmap_np)[0];
    int stateDims[2] = {nPartitions, nReplicates};

    //initialize new data arrays that live in both the CPU and GPU.
    //"Best" and "Working" copies alternately provide "optimizer" and "sampling" views of the Boltzmann machine.
    NumCuda<int> BestStates(2, stateDims);
    NumCuda<int> WrkStates(2,stateDims);
    NumCuda<float> BestEnergies(1, &nReplicates);
    NumCuda<float> WrkEnergies(1, &nReplicates);
    NumCuda<float> AvgEnergy(1, &nBreaks); //CPU only, for tracking energy statistics as annealing progresses

    //set up global halt variable for premature termination of all variables:
    int one = 1;
    NumCuda<int> GlobalHalt(1, &one);
    *GlobalHalt.hd = -1;
    GlobalHalt.CopyHostToDevice();

    // Launch the kernel ============================================================================================================
    int threadsPerBlock = 32;
    int blocksPerGrid = (nReplicates + threadsPerBlock - 1) / threadsPerBlock;
    int perRun = niters/nBreaks;

    for (int i = 0; i < nBreaks; i++){
        int MinIter = i*perRun;
        int MaxIter = MinIter+perRun;
        potts_AllInOne1<<<blocksPerGrid, threadsPerBlock>>>(
            kernels,
            kmap,
            qSizes,
            WrkStates.dd,
            BestStates.dd,
            WrkEnergies.dd,
            BestEnergies.dd,
            MinIter,
            MaxIter,
            nReplicates,
            Tmax,
            Tmin,
            Tperiod,
            e_th,
            GlobalHalt.dd);

        cudaDeviceSynchronize(); //wait for GPU processing to finish
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("kernel execution error: %s\n", cudaGetErrorString(err));
            PyErr_SetString(PyExc_ValueError, "CUDA kernel execution error");
            return NULL;
        }

        EnergySamples.CopyDeviceToHost();
        AvgEnergy(i) = 0;
        for (int j = 0; j<nReplicates; j++)
            AvgEnergy(i) += EnergySamples(j);
        AvgEnergy(i) = AvgEnergy(i)/nReplicates;


    }


    //retrieve results from CUDA memory ===============================================================================================
    //retrieve and package the best state as a numpy array,
    //regardless of if it was a "solution" or not:
    BestEnergies.CopyDeviceToHost()
    BestStates.CopyDeviceToHost()
    int BestReplicate = BestEnergies.argMin();
    NumCuda<int> BestState(1, &nPartitions, *BestStates(BestReplicate, 0)); //initializes from existing non-numpy data. CPU use only
    BestState_np = BestState.ExportNumpy();


    // GlobalHalt.CopyDeviceToHost();
    // if (*GlobalHalt.hd > 0) *GlobalHalt.hd -= 1; //in this case a solution was found


    // cudaMemcpy(&host_gh, global_halt, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Global halt: %i\n",host_gh);
    // if (host_gh > 0) host_gh = host_gh - 1; 

    //retrieve the best result from the GPU:
    int *returned_states = (int *)malloc(nPartitions*sizeof(int));
    int *d_solution_state = BestStates.dd+*GlobalHalt.hd*nPartitions; //new pointer to the best found state:
    cudaError_t err = cudaMemcpy(returned_states, d_solution_state, nPartitions*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("copy back error: %s\n", cudaGetErrorString(err));

    npy_intp const dims [1] = {nPartitions};
    PyObject* array3 = PyArray_SimpleNewFromData(1, dims, NPY_INT, returned_states);
    return array3;


    //free memory:
    // cudaFree(global_halt);  

    //wrap best state in a numpy structure and send it back to python:
    // npy_intp const dims [1] = {nBreaks};
    // PyObject* array3 = PyArray_SimpleNewFromData(1, dims, NPY_FLOAT, h_AvgE);
    // return array3;

}
