#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <stdio.h>
#include <numpy/arrayobject.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


/*
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
__global__ void potts_AllInOne(
    const float *weights,
    int nNHPPs,
    const int *partitions,
    int nPartitions,
    int *working_states,
    int *best_states,
    int niters,
    int nReplicates,
    float T, //temperature of the calculation
    float e_th, //energy threshold to trigger premature termination
    volatile int *global_halt //bit that tells all processes to exit, set by a single process upon finding a 'solution'
    )
{
    
    //index for which replicate this thread is computing; only matters for writing the final best state back to global memory
    int replicate_index = blockDim.x * blockIdx.x + threadIdx.x;

    if (replicate_index + 1 > nReplicates) return;

    //create local references to the state vectors used in this particular thread:
    int *MiWrk = &working_states[replicate_index*nPartitions];
    int *MiBest = &best_states[replicate_index*nPartitions];


    //rng initialization ==========================================================================
    //seed value for random number generator that decides whether a state transition occurs or not:
    int flip_seed = blockDim.x * blockIdx.x + threadIdx.x;
    curandState flipRngState;
    curand_init(flip_seed, 0, 0, &flipRngState);

    //seed value for random number generator that decides which state transitions to propose.
    //by sharing this seed between all threads in a block, memory accesses should be reduced, since the same weights will be requested each time.
    int trial_seed = 10000 + blockDim.x * blockIdx.x;
    curandState trialRngState;
    curand_init(trial_seed, 0, 0, &trialRngState);

    //total energy tracker initialization ===============================================

    //effectively initializes each Potts state to be the highest possible index for that state.
    for (int NHPP = 0; NHPP < nNHPPs; NHPP++)
        MiWrk[partitions[NHPP]] = NHPP;

    //calculate starting total energy:
    float current_e = 0;
    for (int i = 0; i < nPartitions; i++){
        for (int j = 0; j < nPartitions; j++){
            current_e += weights[MiWrk[i]*nNHPPs+MiWrk[j]];
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
        *global_halt = replicate_index+1;
        // __threadfence_system(); //make the global halt value visible to all other threads
    }
            

    //main loop ==================================================================================
    for (int iter = 0; iter < niters; iter++){
        if (*global_halt > 0)
            return; //if one of the threads has found a solution, all threads exit here

        int Mi_proposed = curand(&flipRngState)%nNHPPs;
        curand(&trialRngState);
        int Mi_current = MiWrk[partitions[Mi_proposed]];

        float dE = 0;
        //compute how much the energy would change.
        //assumes no weights between Mi_proposed and Mi_current, otherwise this calculation would be incorrect.
        for (int i = 0; i < nPartitions; i++){
            dE = dE + weights[Mi_proposed*nNHPPs+MiWrk[i]] - weights[Mi_current*nNHPPs+MiWrk[i]];
        }

        float flip_prob = exp(-dE/T);

        if (flip_prob >= curand_uniform(&flipRngState)){
            //update the state:
            MiWrk[partitions[Mi_proposed]] = Mi_proposed;
            //update the energy:
            current_e += dE;
            //and possibly update the lowest energy value:
            if (current_e < lowest_e){
                lowest_e = current_e;
                for (int i = 0; i < nPartitions; i++){
                    MiBest[i] = MiWrk[i];
                }
                if (lowest_e < e_th){
                    *global_halt = replicate_index+1;
                    //threadfence may not be what I needed; maybe just needed to set global_halt as a volatile variable
                    // __threadfence_system(); //make the global halt value visible to all other threads
                }
            }
        }
    }
}




PyObject* PottsGpuFwSolve(PyObject* self, PyObject* args) {

    //init from python inputs =======================================================================================================
    import_array(); //numpy initialization function, otherwise may get seg faults
    PyArrayObject* weights_np, *partitions_np;
    int nReplicates;
    int niters;
    int threadsPerBlock;
    float T;
    float e_th;

    if (!PyArg_ParseTuple(args, "OOiiffi", &weights_np, &partitions_np, &nReplicates, &niters, &T, &e_th, &threadsPerBlock))
        return NULL;

    if (PyArray_NDIM(weights_np) != 2) printf("Weights should have 2 dimensions, not %i\n", PyArray_NDIM(weights_np));
    if (PyArray_NDIM(partitions_np) != 1) printf("Partitions should have 1 dimension, not %i\n", PyArray_NDIM(partitions_np));
    if (PyArray_TYPE(weights_np) != NPY_FLOAT) printf("Weights has type number %i, not %i\n", PyArray_TYPE(weights_np), NPY_FLOAT);
    if (PyArray_TYPE(partitions_np) != NPY_INT) printf("Partitions has type number %i, not %i\n", PyArray_TYPE(partitions_np), NPY_INT);
    
    //Proper way to handle these errors, which I am not doing since it takes a tiny bit more effort:
    // PyErr_SetString(PyExc_ValueError, "arrays must be one-dimensional and of type float");
    // return NULL;

    int nNHPPs = PyArray_SHAPE(weights_np)[0];
 
    //re-frame arrays for easier C++ access:
    int *partitions = (int *)PyArray_DATA(partitions_np);
    float *weights = (float *)PyArray_DATA(weights_np);

    //assume that partitions are in order, so that the last NHPP is a member of the last partition.
    int nPartitions = partitions[nNHPPs-1]+1;

    // set up CUDA memory =========================================================================================================
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    //defining memory:
    int *global_halt = NULL;
    float *d_weights = NULL;
    int *d_wrk_states = NULL;
    int *d_best_states = NULL;
    int *d_partitions = NULL;
    
    //defining memory size, in bytes:
    int sz_w = nNHPPs*nNHPPs*sizeof(float);
    int sz_s = nPartitions*nReplicates*sizeof(int);
    int sz_p = nNHPPs*sizeof(int);

    //allocate memory in the GPU:
    // printf("allocating GPU memory\n");
    err = cudaMalloc((void **)&d_weights, sz_w);
    if (err != cudaSuccess) printf("Weights Malloc error: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **)&d_wrk_states, sz_s);
    if (err != cudaSuccess) printf("Work states Malloc error: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **)&d_best_states, sz_s);
    if (err != cudaSuccess) printf("Best States Malloc error: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **)&d_partitions, sz_p);
    if (err != cudaSuccess) printf("Partitions Malloc error: %s\n", cudaGetErrorString(err));
    err = cudaMalloc((void **)&global_halt, sizeof(int));
    if (err != cudaSuccess) printf("GH Malloc error: %s\n", cudaGetErrorString(err));

    //copy inputs to the GPU:
    // printf("Copying data to GPU\n");
    err = cudaMemcpy(d_weights, weights, sz_w, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Weights memcopy error: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_partitions,  partitions,  sz_p, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("Partitions memcopy error: %s\n", cudaGetErrorString(err));
    int host_gh = 0;
    err = cudaMemcpy(global_halt, &host_gh, sizeof(int), cudaMemcpyHostToDevice);//initialized flag
    if (err != cudaSuccess) printf("GH memcopy error: %s\n", cudaGetErrorString(err));

    // Launch the kernel ============================================================================================================
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
    // printf("GPU has %i SMs\n", deviceProp.multiProcessorCount);
    int blocksPerGrid = (nReplicates + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    potts_AllInOne<<<blocksPerGrid, threadsPerBlock>>>(
        d_weights,
        nNHPPs,
        d_partitions,
        nPartitions,
        d_wrk_states,
        d_best_states,
        niters,
        nReplicates,
        T,
        e_th,
        global_halt);

    cudaDeviceSynchronize(); //wait for GPU processing to finish
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel execution error: %s\n", cudaGetErrorString(err));
        PyErr_SetString(PyExc_ValueError, "CUDA kernel execution error");
        return NULL;
    }



    //retreive results from CUDA memory ===============================================================================================
    //retrieve index of the best result:
    cudaMemcpy(&host_gh, global_halt, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Global halt: %i\n",host_gh);
    if (host_gh > 0) host_gh = host_gh - 1; //in this case a solution was found

    //retrieve the best result from the GPU:
    int *returned_states = (int *)malloc(nPartitions*sizeof(int));
    int *d_solution_state = d_best_states+host_gh*nPartitions; //new pointer to the best found state:
    err = cudaMemcpy(returned_states, d_solution_state, nPartitions*sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("copy back error: %s\n", cudaGetErrorString(err));

    //free memory:
    cudaFree(d_weights);
    cudaFree(d_wrk_states);
    cudaFree(d_best_states);
    cudaFree(d_partitions);
    cudaFree(global_halt);

    //wrap best state in a numpy structure and send it back to python:
    npy_intp const dims [1] = {nPartitions};
    PyObject* array3 = PyArray_SimpleNewFromData(1, dims, NPY_INT, returned_states);
    return array3;

}

/*
//"main" routine for one-line compile, run and test using "nvcc -run potts_per_thread.cu"
int main(){
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    //initialize problem in c code space:
    int nNHPPs = 9;
    int nPartitions = 3;
    int nReplicates = 4;
    int niters = 10;
    float T = 1;

    //defining memory:
    float weights[nNHPPs*nNHPPs] = {
        0,0,0, 1,0,0, 1,0,0,
        0,0,0, 0,1,0, 0,1,0,
        0,0,0, 0,0,1, 0,0,1,

        1,0,0, 0,0,0, 1,0,0,
        0,1,0, 0,0,0, 0,1,0,
        0,0,1, 0,0,0, 0,0,1,

        1,0,0, 1,0,0, 0,0,0,
        0,1,0, 0,1,0, 0,0,0,
        0,0,1, 0,0,1, 0,0,0,
    };
    float *d_weights = NULL;
    int returned_states[nPartitions*nReplicates];
    int *d_wrk_states = NULL;
    int *d_best_states = NULL;
    bool *global_halt = NULL;
    int partitions[nNHPPs] = {0,0,0,1,1,1,2,2,2};
    int *d_partitions = NULL;

    //defining memory size, in bytes:
    int sz_w = nNHPPs*nNHPPs*sizeof(float);
    int sz_s = nPartitions*nReplicates*sizeof(int);
    int sz_p = nNHPPs*sizeof(int);

    //allocate memory in the GPU:
    printf("allocating GPU memory\n");
    err = cudaMalloc((void **)&d_weights, sz_w);
    err = cudaMalloc((void **)&d_wrk_states, sz_s);
    err = cudaMalloc((void **)&d_best_states, sz_s);
    err = cudaMalloc((void **)&d_partitions, sz_p);
    err = cudaMalloc((void **)&global_halt, sizeof(bool));

    //copy inputs to the GPU:
    printf("Copying data to GPU\n");
    cudaMemcpy(d_weights, weights, sz_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_partitions,  partitions,  sz_p, cudaMemcpyHostToDevice);
    *global_halt = false;

    // Launch the kernel
    int threadsPerBlock = 2;
    int blocksPerGrid = 2;//(nNHPPs + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    potts_AllInOne<<<blocksPerGrid, threadsPerBlock>>>(
        d_weights,
        nNHPPs,
        d_partitions,
        nPartitions,
        d_wrk_states,
        d_best_states,
        niters,
        nReplicates,
        T,
        -1e9,
        global_halt);

    cudaDeviceSynchronize(); //wait for GPU processing to finish
    err = cudaGetLastError();
    if (err != cudaSuccess) printf("kernel execution error: %s\n", cudaGetErrorString(err));

    //retrieve the result from the GPU:
    err = cudaMemcpy(returned_states, d_best_states, sz_s, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("copy back error: %s\n", cudaGetErrorString(err));

    //print out the result:
    for (int j = 0; j < nReplicates; j++){
        for (int i = 0; i < nPartitions; i++)
            printf("%i, ", returned_states[j*nPartitions+i]);
        printf("\n");
    }

    //free memory:
    cudaFree(d_weights);
    cudaFree(d_wrk_states);
    cudaFree(d_best_states);
    cudaFree(d_partitions);


    return 0;
}
*/