#include <stdio.h>
#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "NumCuda.hpp"
#include "PieceWiseLinear.cpp"



/*
cuda kernel for sampling a single replicate of a Potts model.

Cooperative model: assumes that all cooperating threads are in a single warp (block size = warp size),
And thus is automatically synchronized.

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
__global__ void EfirstCooperative(
    NumCuda<float> weights,
    NumCuda<int> partitions,
    NumCuda<int> working_states,
    NumCuda<int> best_states,
    NumCuda<float> working_energies,
    NumCuda<float> best_energies,
    NumCuda<float> potential_energies,//shared amoung a cooperative group, but not interactive
    NumCuda<float> per_thread_proposals, //mediates interactions between cooperating threads
    int MinIter,
    int MaxIter,
    PieceWiseLinear PwlTemp,
    float e_th, //energy threshold to trigger premature termination
    volatile int *global_halt //bit that tells all processes to exit, set by a single process upon finding a 'solution'.
    //super important that this is declared volatile, so that it is always retreived from global memory, 
    )
{
    
    //index for which replicate this thread is computing; only matters for writing the final best state back to global memory
    #ifdef __CUDA_ARCH__
        int replicate_index = blockIdx.x;
        int thread_index = threadIdx.x;
        int rngIndex = blockDim.x * blockIdx.x + threadIdx.x;
        int coopSize = blockDim.x; //number of threads in a block that are all working together
        // printf("I am thread %i working on replicate %i with %i other threads\n", thread_index, replicate_index, coopSize-1);
    #else
        int replicate_index = 0;
        int thread_index = 0;
        int rngIndex = 0;
        int coopSize = 1;
    #endif
    
    //get some dimensional values from the passed NumCuda arrays:
    int nReplicates = working_energies.dims[0];
    int nPartitions = working_states.dims[1];
    int nNHPPs = weights.dims[0];
    if (replicate_index + 1 > nReplicates) return;

    //create local references to the state vectors used in this particular thread:
    int *MiWrk = &working_states(replicate_index, 0);
    int *MiBest = &best_states(replicate_index, 0);
    float *EParts = &potential_energies(replicate_index, 0);

    //rng initialization.  Different libraries are used, depending on if we're compiling for host or gpu
    #ifdef __CUDA_ARCH__
        curandState RngState;
        curand_init(rngIndex, MinIter, 0, &RngState); //use thread index number and starting iteration to seed the rng
        #define RngUniform() curand_uniform(&RngState)
        #define RngInteger() curand(&RngState)
        // printf("running in Cuda kernel\n");

    #else
        std::mt19937 RngGenerator(rngIndex + MinIter*1000);
        std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);
        #define RngUniform() uniform_distribution(RngGenerator)
        #define RngInteger() RngGenerator()

    #endif

    
    // =================================================================init states
    if (thread_index == 0){ //first thread is responsible for state initialization
        if (MinIter == 0){
            //then states should be newly initialized.
            //effectively initializes each Potts state to be the highest possible index for that state:
            for (int NHPP = 0; NHPP < nNHPPs; NHPP++){
                MiWrk[partitions(NHPP)] = NHPP;
                MiBest[partitions(NHPP)] = NHPP;
                // printf("I'm thread #%i, and I just set MiWrk[%i]=%i\n", thread_index, partitions(NHPP), NHPP);
            }
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
    }
    // return;
    __syncthreads();

    // =================================================================init energy parts
    for (int NHPP = thread_index; NHPP < nNHPPs; NHPP += coopSize){
        EParts[NHPP] = 0;

        for (int partition = 0; partition < nPartitions; partition++){
            // if (MiWrk[partition] >= nNHPPs) printf("Error, MiWrk[%i] = %i\n", partition, MiWrk[partition]);
            // if (MiWrk[NHPP] >= nNHPPs) printf("Error, MiWrk[%i] = %i\n", NHPP, MiWrk[NHPP]);
            EParts[NHPP] += weights(MiWrk[partition], NHPP);
            //this loop order is inneffcient, but OK since it only runs once.
            //in main code part, there is no inner loop, so the outer loop can run efficiently.
        }
        // printf("I am thread %i, and I just initialized NHPP #%i\n", thread_index, NHPP);
    }

    
    // ==================================================================init total energy
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
            

    // ====================================================================================== main loop
    //at each loop iteration, code for logging changes from the previous NHPP switch and code for proposing the next switch are interleaved.
    //the start of the loop assumes that it needs to log a change from the previous loop, so on the first iteration we need a dummy initialization.
    //Why? because this ordering allows possible parallelization of independent serial sections.
    int newNHPP = MiWrk[0];//the new one, by definition, is the old one
    int oldNHPP = MiWrk[0];
    float dE = 0;
    for (int iter = MinIter; iter < MaxIter; iter++){

        if (*global_halt > 0)
            break; //if one of the threads has found a solution, all threads exit here


        //==========================================================================propose NHPP changes (parallel)
        //parallel component of algorithm: each thread is responsible for updating the energy of a subset of NHPPs,
        //and also calculating probabilties and proposing one NHPP as the first-to-fire for this round.
        int ProposedNHPP;
        float ProposedPriority = 1e9;
        float T = PwlTemp.interp(iter);
        // float threadE = 0;
        for (int NHPP = thread_index; NHPP < nNHPPs; NHPP += coopSize){
            EParts[NHPP] += weights(newNHPP, NHPP);
            EParts[NHPP] -= weights(oldNHPP, NHPP);//Partial energies update finished
            // threadE += EParts[NHPP];
            if (NHPP != MiWrk[partitions(NHPP)]){//start stochastic NHPP-style selection, but only for NHPPs that do not already have latch control
                float proposed_dE = EParts[NHPP]-EParts[MiWrk[partitions(NHPP)]];
                float NhppRate = exp(-proposed_dE/T);
                float NewPriority = -log(RngUniform())/NhppRate; 
                // printf("NewPriority of NHPP %i = %.2f\n", NHPP, NewPriority);
                if (NewPriority < ProposedPriority){
                    ProposedPriority = NewPriority;
                    ProposedNHPP = NHPP;
                }
            }
        }
        per_thread_proposals(replicate_index, thread_index, 0) = ProposedPriority;
        per_thread_proposals(replicate_index, thread_index, 1) = ProposedNHPP;
        // per_thread_proposals(replicate_index, thread_index, 2) = threadE;

        __syncthreads();

        //bottleneck: first thread selects which NHPP will be the next one to go active
        if (thread_index == 0){
            // ===================================================================================== finalize next new NHPP
            //proposed priority for thread zero is already available here, so we'll start with that:
            int ProposedThread = 0;
            for (int thrd = 1; thrd < coopSize; thrd++){
                if (per_thread_proposals(replicate_index, thrd, 0) < ProposedPriority){
                    // printf("Thread %i's proposal of NHPP #%i with a score of %.2f is the best I've seen so far\n", thrd, per_thread_proposals(replicate_index, thrd, 1), per_thread_proposals(replicate_index, thrd, 0));
                    ProposedPriority = per_thread_proposals(replicate_index, thrd, 0);
                    ProposedThread = thrd;
                }
            }
            //write back the final selected NHPP for all threads to see:
            per_thread_proposals(replicate_index, 0, 0) = per_thread_proposals(replicate_index, ProposedThread, 1);
        }

        //while at the same time (or not, depending on CUDA version and hardware and number or worker threads),
        //the update proposed in the last cycle is logged
            // ===================================================================================== update global energy and states
            //update energies:
            
        if (thread_index == coopSize-1){
            // printf("I'm thread %i and I'm about to set MiWrk[%i] = %i\n", thread_index, partitions(newNHPP), newNHPP);
            MiWrk[partitions(newNHPP)] = newNHPP;
            current_e += dE;
            if (current_e < lowest_e){
                lowest_e = current_e;
                for (int i = 0; i < nPartitions; i++){
                    MiBest[i] = MiWrk[i];
                }
                if (lowest_e < e_th){
                    *global_halt = 1;
                }
            }

        }
        __syncthreads();

        //the implications of these changes will be taken into account in the next loop iteration
        newNHPP = per_thread_proposals(replicate_index, 0, 0);
        oldNHPP = MiWrk[partitions(newNHPP)];
        dE = EParts[newNHPP] - EParts[oldNHPP];

    }

    __syncthreads(); //make sure all threads are done before reformatting the states

    //last thread needs to update energies in global memory on its way out,
    //and also re-format states for outside consumption
    if (thread_index == coopSize-1){ 

        working_energies(replicate_index) = current_e;
        best_energies(replicate_index) = lowest_e;

        int last_partition = -1;
        for (int NHPP = 0; NHPP < nNHPPs; NHPP++){
            if (partitions(NHPP) > last_partition){
                last_partition = partitions(NHPP);
                MiWrk[last_partition] -= NHPP;
                MiBest[last_partition] -= NHPP;
            }
        }
    }

    return;
}