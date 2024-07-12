#include <stdio.h>
#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "NumCuda.hpp"
#include "PieceWiseLinear.cpp"


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

__host__ __device__ void PottsFullWeightsHost(
    NumCuda<float> &weights, //if these are not passed by reference, the pointers within cannot be accesssed... I don't know why
    NumCuda<int> &partitions,
    NumCuda<int> &working_states,
    NumCuda<int> &best_states,
    NumCuda<float> &working_energies,
    NumCuda<float> &best_energies,
    int minIter,
    int maxIter,
    PieceWiseLinear PwlTemp,
    float e_th, //energy threshold to trigger premature termination
    volatile int *global_halt //bit that tells all processes to exit, set by a single process upon finding a 'solution'
    )
{
    
    //index for which replicate this thread is computing; only matters for writing the final best state back to global memory
    #ifdef __CUDA_ARCH__
        int replicate_index = blockDim.x * blockIdx.x + threadIdx.x;
    #else
        int replicate_index = 0;
    #endif
    
    //get some dimensional values from the passed NumCuda arrays:
    int nReplicates = working_energies.dims[0];
    int nNHPPs = weights.dims[0];
    int nPartitions = working_states.dims[1];
    if (replicate_index + 1 > nReplicates) return;

    //create local references to the state vectors used in this particular thread:
    int *MiWrk = &working_states(replicate_index, 0);
    int *MiBest = &best_states(replicate_index, 0);

    //rng initialization.  Different libraries are used, depending on if we're compiling for host or gpu
    #ifdef __CUDA_ARCH__
        curandState RngState;
        curand_init(replicate_index, minIter, 0, &RngState); //use thread index number and starting iteration to seed the rng
        #define RngUniform() curand_uniform(&RngState)
        #define RngInteger() curand(&RngState)
        // printf("running in Cuda kernel\n");

    #else
        std::mt19937 RngGenerator(replicate_index + minIter*1000);
        std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);
        #define RngUniform() uniform_distribution(RngGenerator)
        #define RngInteger() RngGenerator()

    #endif
    
    //total energy tracker initialization ===============================================

    if (minIter == 0){
        //then states should be newly initialized.
        //effectively initializes each Potts state to be the highest possible index for that state:
        for (int NHPP = 0; NHPP < nNHPPs; NHPP++){
            MiWrk[partitions(NHPP)] = NHPP;
            MiBest[partitions(NHPP)] = NHPP;}
    }
    else{
        //since a non-standard state format is used,
        //we need to convert the standard state (which is stored between runs)
        //to the internally-used non-standard state
        int last_partition = -1;
        for (int NHPP = 0; NHPP < nNHPPs; NHPP++){
            if (partitions(NHPP) > last_partition){
                last_partition = partitions(NHPP);
                MiWrk[last_partition] += NHPP;
                MiBest[last_partition] += NHPP;
            }
        }
    }
    
    // return;

    //calculate starting total energy:
    float current_e = 0;
    float lowest_e = 0;
    for (int i = 0; i < nPartitions; i++){
        for (int j = 0; j < nPartitions; j++){
            current_e += weights(MiWrk[i], MiWrk[j]);
            lowest_e += weights(MiBest[i], MiBest[j]);
            //weights contribute to the active energy when both sides of the weights are selected.
        }
    }
    current_e = current_e / 2;
    lowest_e = lowest_e / 2;

    if (lowest_e < e_th){
        *global_halt = replicate_index+1;
    }
            

    //main loop ==================================================================================
    for (int iter = minIter; iter < maxIter; iter++){
        if (*global_halt > 0)
            break; //if one of the threads has found a solution, all threads exit here

        int Mi_proposed = RngInteger()%nNHPPs;
        int Mi_current = MiWrk[partitions(Mi_proposed)];

        float dE = 0;
        //compute how much the energy would change.
        //assumes no weights between Mi_proposed and Mi_current, otherwise this calculation would be incorrect.
        for (int i = 0; i < nPartitions; i++){
            dE = dE + weights(Mi_proposed, MiWrk[i]) - weights(Mi_current, MiWrk[i]);
        }

        float T = PwlTemp.interp(iter);
        float flip_prob = exp(-dE/T);

        if (flip_prob >= RngUniform()){
            //update the state:
            MiWrk[partitions(Mi_proposed)] = Mi_proposed;
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
                }
            }
        }
    }
    working_energies(replicate_index) = current_e;
    best_energies(replicate_index) = lowest_e;

    //convert internal state format back to general state format:
    int last_partition = -1;
    for (int NHPP = 0; NHPP < nNHPPs; NHPP++){
        if (partitions(NHPP) > last_partition){
            last_partition = partitions(NHPP);
            MiWrk[last_partition] -= NHPP;
            MiBest[last_partition] -= NHPP;
        }
    }
}



// ============================================================wrapper to call main code as a GPU kernel
__global__ void PottsFullWeightsDevice(
    NumCuda<float> weights,
    NumCuda<int> partitions,
    NumCuda<int> working_states,
    NumCuda<int> best_states,
    NumCuda<float> working_energies,
    NumCuda<float> best_energies,
    int minIter,
    int maxIter,
    PieceWiseLinear PwlTemp,
    float e_th, //energy threshold to trigger premature termination
    volatile int *global_halt //bit that tells all processes to exit, set by a single process upon finding a 'solution'
    )
{
    // return;
    PottsFullWeightsHost(
        weights, partitions,
        working_states, best_states,
        working_energies, best_energies,
        minIter, maxIter,
        PwlTemp, e_th, global_halt
    );
}