#include <stdio.h>
#include <random>
#include <thread>
#include <barrier>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "NumCuda.hpp"
#include "PieceWiseLinear.cpp"

template <typename Annealable> __global__  void AnnealingCoreGpu(
    Annealable task,
    NumCuda<int> working_states,
    NumCuda<int> best_states,
    NumCuda<float> working_energies,
    NumCuda<float> best_energies,
    NumCuda<float> commspace, //for communication between cooperating threads; use however seen fit.
    PieceWiseLinear PwlTemp, //for calculating the annealing temperature to use.
    int OptsPerThrd,
    bool TakeAllActions,
    int MinIter,
    int MaxIter,
    float e_th, //energy threshold to trigger premature termination
    volatile int *TaskDone //bit that tells all processes to exit, set by a single process upon finding a 'solution'.
    //super important that this is declared volatile, so that it is always retreived from global memory, 
){
    int ThrdIndx = blockDim.x * blockIdx.x + threadIdx.x;
    #define SYNC_REPLICATE() __syncthreads()
    curandState RngState;
    curand_init(ThrdIndx, MinIter, 0, &RngState); //use thread index number and starting iteration to seed the rng
    #define RngUniform() curand_uniform(&RngState)
    #define RngInteger() curand(&RngState)

    // #include "functionBody.cpp"
    int nThrds = commspace.dims[1]; //threads per replicate
    int nReplicates = commspace.dims[0];
    int replicate_index = ThrdIndx/nThrds;
    int Thrd = ThrdIndx%nThrds; //thread index within each replicate
    if (replicate_index + 1 > nReplicates) return;

    // Annealable* task = &taskIn;

    //create local references to the state vectors used in this particular thread:
    int *MiWrk = &working_states(replicate_index, 0);
    int *MiBest = &best_states(replicate_index, 0);

    task.SetIdentity(Thrd, nThrds, replicate_index, MiWrk, MiBest);
    task.BeginEpoch(MinIter);

    int iter_inc = 1;
    if (TakeAllActions) iter_inc = nThrds;

    for (int iter = MinIter; iter < MaxIter; iter+=iter_inc){
        if (*TaskDone > 0) break;
        SYNC_REPLICATE();
        //each thread selects the best action within its subset:
        float T = PwlTemp.interp(iter);

        int action = -1; //start with default Null action, with dE=0.
        float BestPriority = -log2(RngUniform()); //for keeping track of which action will be best
        for (int i=0; i<OptsPerThrd;i++){
            int possible_action = RngInteger()%task.NumActions;
            float possible_dE = task.GetActionDE(possible_action);

            //switched signs in order to convert divisions to multiplications:
            float NhppRate = exp2(possible_dE/T); //exp2(-possible_dE/T); 
            float NewPriority = -log2(RngUniform())*NhppRate; ///NhppRate;
            // float NewPriority = possible_dE; //short circuit the NHPP

            if (NewPriority < BestPriority){
                BestPriority = NewPriority;
                action = possible_action;
            }
        }
        commspace(replicate_index, Thrd, 0) = BestPriority;
        commspace(replicate_index, Thrd, 1) = action;

        //synchronize threads, so that all threads can see the action proposals from the other threads.
        //this may sync with other threads that are not working cooperatively, too;
        //while that might be undesirable for performance, it is functionally okay.
        SYNC_REPLICATE();

        //all threads simultaneously decide which actions to take on their annealable object, although this is redundant.
        //They should all be doing exactly the same thing, if they are not, that would be bad.
        if (TakeAllActions){
            //all threads take the same nThrds actions
            for (int i=0;i<nThrds;i++){
                action = commspace(replicate_index, i, 1);
                if (action >= 0)
                    task.TakeAction_tic(action);
                SYNC_REPLICATE();
                task.TakeAction_toc(action);
                SYNC_REPLICATE();
            }
        } else {
            //finds the single best action from all threads, and takes that action.
            for (int i=0;i<nThrds;i++){
                if (commspace(replicate_index, i, 0) <= BestPriority){
                    BestPriority = commspace(replicate_index, i, 0);
                    action = commspace(replicate_index, i, 1);
                }
            }
            if (action >= 0)
                task.TakeAction_tic(action);
            SYNC_REPLICATE();
            task.TakeAction_toc(action);
        }
        if (task.lowest_e < e_th) *TaskDone = 1; 
    }

    working_energies(replicate_index) = task.current_e;
    best_energies(replicate_index) = task.lowest_e;
    task.FinishEpoch();
}

// #undef RngUniform
// #undef RngInteger
// #undef SYNC_REPLICATE

template <typename Annealable> __host__  void AnnealingCoreCpu(
    Annealable &&task,
    NumCuda<int> &&working_states,
    NumCuda<int> &&best_states,
    NumCuda<float> &&working_energies,
    NumCuda<float> &&best_energies,
    NumCuda<float> &&commspace, //for communication between cooperating threads; use however seen fit.
    PieceWiseLinear &&PwlTemp, //for calculating the annealing temperature to use.
    int OptsPerThrd,
    // bool TakeAllActions,
    int MinIter,
    int MaxIter,
    float e_th, //energy threshold to trigger premature termination
    int ThrdIndx,
    // std::barrier<> *sync_point,
    volatile int *TaskDone //bit that tells all processes to exit, set by a single process upon finding a 'solution'.
    //super important that this is declared volatile, so that it is always retreived from global memory, 
){

    // #define SYNC_REPLICATE() sync_point->arrive_and_wait()
    std::mt19937 RngGenerator(ThrdIndx + MinIter*1000);
    std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);
    // #define RngUniform() uniform_distribution(RngGenerator)
    // #define RngInteger() RngGenerator()

    // #include "functionBody.cpp"
    int nThrds = 1;//commspace.dims[1]; //threads per replicate
    int nReplicates = commspace.dims[0];
    int replicate_index = ThrdIndx;///nThrds;
    int Thrd = ThrdIndx%nThrds; //thread index within each replicate
    if (replicate_index + 1 > nReplicates) return;

    // Annealable* task = &taskIn;

    //create local references to the state vectors used in this particular thread:
    int *MiWrk = &working_states(replicate_index, 0);
    int *MiBest = &best_states(replicate_index, 0);

    task.SetIdentity(Thrd, nThrds, replicate_index, MiWrk, MiBest);
    task.BeginEpoch(MinIter);

    for (int iter = MinIter; iter < MaxIter; iter++){
        if (*TaskDone > 0) break;
        //each thread selects the best action within its subset:
        float T = PwlTemp.interp(iter);

        int action = -1; //start with default Null action, with dE=0.
        float BestPriority = -log2(uniform_distribution(RngGenerator)); //for keeping track of which action will be best
        for (int i=0; i<OptsPerThrd;i++){
            int possible_action = RngGenerator()%task.NumActions;
            float possible_dE = task.GetActionDE(possible_action);

            //switched signs in order to convert divisions to multiplications:
            float NhppRate = exp2(possible_dE/T); //exp2(-possible_dE/T); 
            float NewPriority = -log2(uniform_distribution(RngGenerator))*NhppRate; ///NhppRate;

            if (NewPriority < BestPriority){
                BestPriority = NewPriority;
                action = possible_action;
            }
        }
        if (action >= 0) task.TakeAction_tic(action);
        task.TakeAction_toc(action);

        if (task.lowest_e < e_th) *TaskDone = 1; 
    }

    working_energies(replicate_index) = task.current_e;
    best_energies(replicate_index) = task.lowest_e;
    task.FinishEpoch();
}



//core annealing is further wrapped in template dispatch functions, 
//which supports sudo-multi-dispatch function calling in the main routine.
//The main routine gets pointers to these functions,
//which all have the same call signature even though they are individually compiled for each type of annealable object.
//Since this requires passing the task via pointer, we can't directly dispatch the kernel for either CPU or GPU,
//since then all threads would have the same task copy/the GPU would have a pointer to host memory.

//defines format of the dispatch functions, for creating function pointers more easily
using DispatchFunc = void(
        void*, //the task, which must be specified as a void pointer since it changes
        NumCuda<int> &, NumCuda<int> &, //working states, best states
        NumCuda<float> &, NumCuda<float> &, //working energies, best energies
        NumCuda<float> &, //communal space memory for interthread cooperation
        PieceWiseLinear, int, bool, //annealing temperature specification, OptsPerThrd, TakeAllActions
        int, int, //min iter, max iter
        float, //termination threshold energy
        NumCuda<int> &//global halt
        );


template <typename Annealable> void GpuDispatch(
        void* void_task, //the task, which must be specified as a void pointer since it changes
        NumCuda<int> &WrkStates,
        NumCuda<int> &BestStates,
        NumCuda<float> &WrkEnergies,
        NumCuda<float> &BestEnergies,
        NumCuda<float> &commspace,
        PieceWiseLinear PwlTemp,
        int OptsPerThrd,
        bool TakeAllActions,
        int MinIter,
        int MaxIter,
        float e_th,
        NumCuda<int> &global_halt
        ){

    Annealable task = *(Annealable*)void_task; //converts the void pointer to an Annealable pointer, dereferences it, and makes a copy

    //GPU parallelism parameters:
    int nWorkers = commspace.dims[1];
    int nReplicates = commspace.dims[0];
    int threadsPerBlock;
    if (nWorkers < 64) threadsPerBlock = int(64/nWorkers)*nWorkers; //targets less than, but close to, 64 threads per block
    else threadsPerBlock = nWorkers;
    int ReplicatesPerBlock = threadsPerBlock/nWorkers;
    int blocksPerGrid = (nReplicates + ReplicatesPerBlock - 1) / ReplicatesPerBlock;
    // printf("Launching kernel with %i blocks of %i threads\n", blocksPerGrid, threadsPerBlock);
    AnnealingCoreGpu<Annealable><<<blocksPerGrid, threadsPerBlock>>>(
                task,
                WrkStates,         BestStates,
                WrkEnergies,       BestEnergies,
                commspace,
                PwlTemp,           OptsPerThrd, TakeAllActions,
                MinIter,           MaxIter,
                e_th,
                global_halt.dd);

    //GPU cleanup and result memory synchronization:
    cudaDeviceSynchronize(); //wait for GPU processing to finish
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Last error after kernel execution: %s\n", cudaGetErrorString(cudaGetLastError()));
    BestStates.CopyDeviceToHost();
    WrkStates.CopyDeviceToHost();
    BestEnergies.CopyDeviceToHost();
    WrkEnergies.CopyDeviceToHost();

}

template <typename Annealable> void CpuDispatch(
        void* void_task, //the task, which must be specified as a void pointer since it changes
        NumCuda<int> &WrkStates,
        NumCuda<int> &BestStates,
        NumCuda<float> &WrkEnergies,
        NumCuda<float> &BestEnergies,
        NumCuda<float> &commspace,
        PieceWiseLinear PwlTemp,
        int OptsPerThrd,
        bool TakeAllActions,
        int MinIter,
        int MaxIter,
        float e_th,
        NumCuda<int> &GlobalHalt
        ){

    Annealable task = *(Annealable*)void_task; //converts the void pointer to an Annealable pointer, dereferences it, and makes a copy

    int nWorkers = commspace.dims[1];
    int nReplicates = commspace.dims[0];
    int nThreads = nWorkers*nReplicates;

    std::vector<std::thread> workers;

    // std::barrier<> * b; 
    //tried using barriers to make CPU thread worker groups. 
    //Unfortunately CPU Thread scheduling makes this very slow, so I've removed it.
    //Now different threads just compute their own replicates.

    for (int thread_num = 0; thread_num < nThreads; thread_num++){
        workers.push_back(std::thread(AnnealingCoreCpu<Annealable>,
                    task,
                    WrkStates,         BestStates,
                    WrkEnergies,       BestEnergies,
                    commspace,
                    PwlTemp,           OptsPerThrd, //TakeAllActions,
                    MinIter,           MaxIter,
                    e_th,
                    thread_num,
                    // b,
                    GlobalHalt.hd));
    }

    //wait for all the threads to finish
    for (std::thread &t: workers) {
      if (t.joinable()) {
        t.join();
      }
    }

}