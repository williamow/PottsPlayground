#include <stdio.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

//definitions for actual NVCC compilation within dual-use headers:
#define __h__ __host__
#define __d__ __device__

#include "NumCuda.h"
#include "PieceWiseLinear.h"
 
#include "Annealables.h"
#include "TspAnnealable.h"
#include "IsingAnnealable.h"
#include "PottsJitAnnealable.h"
#include "PottsPrecomputeAnnealable.h"

#include "Cores.h"

template class NumCuda<int>;
template class NumCuda<float>;

bool GetGpuAvailability(){
	bool GPU_AVAIL;
	int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No GPUs detected, defaulting to CPU-only operation.\n");
        GPU_AVAIL = false;
        // GPU_THREADS = 0;
    } else {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, /*deviceCount=*/ 0);
        float mem_gb = static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024);
        printf("GPU %s is available, with %i streaming multiprocessors and %.2f GB memory \n", deviceProp.name, deviceProp.multiProcessorCount, mem_gb);
        GPU_AVAIL = true;
        // GPU_THREADS = deviceProp.multiProcessorCount*deviceProp.maxThreadsPerMultiProcessor;
    }
    return GPU_AVAIL;
}

template <typename Annealable> __global__  void AnnealingCoreGpu(
	Annealable task,
	NumCuda<int> working_states,
	NumCuda<int> best_states,
	NumCuda<float> working_energies,
	NumCuda<float> best_energies,
	NumCuda<float> ActArb, //for communication between cooperating threads; use however seen fit.
	PieceWiseLinear PwlTemp, //for calculating the annealing temperature to use.
	int nOptions,
	int nActions,
	long MinIter,
	long MaxIter,
	float e_th, //energy threshold to trigger premature termination
	volatile int *TaskDone //bit that tells all processes to exit, set by a single process upon finding a 'solution'.
	//super important that this is declared volatile, so that it is always retreived from global memory, 
){
	int ThrdIndx = blockDim.x * blockIdx.x + threadIdx.x;

	curandStateMRG32k3a_t RngState;
	curand_init(MinIter, ThrdIndx, 0, &RngState); //use thread index number and starting iteration to seed the rng
	#define RngUniform() curand_uniform_double(&RngState)
	#define RngInteger() curand(&RngState)

	int nThrds = blockDim.x;//ActArb.dims[1]; //threads per replicate //shit this does not work any more. Ugghghghgh
	int nReplicates = ActArb.dims[0];
	int replicate_index = blockIdx.x;//ThrdIndx/nThrds;
	int Thrd = threadIdx.x; //thread index within each replicate
	if (replicate_index + 1 > nReplicates) return;

	//create local references to the state vectors used in this particular thread:
	int *MiWrk = &working_states(replicate_index, 0);
	int *MiBest = &best_states(replicate_index, 0);

	task.SetIdentity(Thrd, nThrds, replicate_index, MiWrk, MiBest);
	task.BeginEpoch(MinIter);

	// int iter_inc = nActions ? nThrds : 1;
	bool FullParallel = (nOptions >= task.NumActions);
	
	for (long iter = MinIter; iter < MaxIter; iter+=nActions){
		if ((iter+nActions)%1000000 < iter%1000000){
			//condition to re-init task, to smooth over errors that might otherwise accumulate
			task.FinishEpoch();
			task.BeginEpoch(iter);
		}


		if (*TaskDone > 0) break;
		__syncthreads();
		
		double beta = 1./PwlTemp.interp(iter);

		for (int act = Thrd; act<nActions; act+=nThrds)
			ActArb(replicate_index, act, 0) = -log(RngUniform());
			//null priorities are initialized here since it uses device-dependent rng code

		ActionArbitrator Arb(&ActArb, replicate_index);

		for (int i=Thrd; i<nOptions; i+=nThrds){
			int NewOption = FullParallel ? i : RngInteger()%task.NumActions;
			if (NewOption >= task.NumActions) break; //esp. for FullParallel mode, where OpsPerThread is not the right number of steps

			float possible_dE = task.GetActionDE(NewOption);

			//switched signs in order to convert divisions to multiplications:
			float NhppRate = exp(possible_dE*beta); //exp2(-possible_dE/T); 
			float NewPriority = -log(RngUniform())*NhppRate; ///NhppRate;

			//omg this is such gross parallelism. sad to have in gpu code
			Arb.AddOption(NewOption, NewPriority); //if the new option has a higher priority, it will be added to ActArb memory
		}

		//synchronize threads, so that all threads can see the action proposals from the other threads.
		//this may sync with other threads that are not working cooperatively, too;
		//while that might be undesirable for performance, it is functionally okay.
		__syncthreads();

		//skip the last TakeAction, 
		//so the last state and its dwell time can be recorded together for technically correct sampling probabilities
		if (iter + nActions >= MaxIter) break;

		for (int act = 0; act<nActions; act++){
			int action = ActArb(replicate_index, act, 1);
			if (action >= 0) task.TakeAction_tic(action);
			__syncthreads();
			task.TakeAction_toc(action);
			__syncthreads();
		}

		if (task.lowest_e < e_th) *TaskDone = 1; 
	}

	task.FinishEpoch();
	working_energies(replicate_index) = task.current_e;
	best_energies(replicate_index) = task.lowest_e;
}

//core annealing is further wrapped in template dispatch functions, 
//which supports sudo-multi-dispatch function calling in the main routine.
//The main routine gets pointers to these functions,
//which all have the same call signature even though they are individually compiled for each type of annealable object.
//Since this requires passing the task via pointer, we can't directly dispatch the kernel for either CPU or GPU,
//since then all threads would have the same task copy/the GPU would have a pointer to host memory.
template <typename Annealable> void GpuDispatch(
		void* void_task, //the task, which must be specified as a void pointer since it changes
		NumCuda<int> &WrkStates,
		NumCuda<int> &BestStates,
		NumCuda<float> &WrkEnergies,
		NumCuda<float> &BestEnergies,
		NumCuda<float> &ActArb,
		NumCuda<int> &RealFlips,
		PieceWiseLinear PwlTemp,
		int nOptions,
		int nActions,
		int nWorkers,
		long MinIter,
		long MaxIter,
		float e_th,
		NumCuda<int> &global_halt
		){

	Annealable task = *(Annealable*)void_task; //converts the void pointer to an Annealable pointer, dereferences it, and makes a copy

	//GPU parallelism parameters:
	// int nWorkers = ActArb.dims[1];
	int nReplicates = ActArb.dims[0];
	int threadsPerBlock = nWorkers;
	// if (nWorkers < 64) threadsPerBlock = int(64/nWorkers)*nWorkers; //targets less than, but close to, 64 threads per block
	// else threadsPerBlock = nWorkers;
	int ReplicatesPerBlock = 1;//threadsPerBlock/nWorkers;
	int blocksPerGrid = (nReplicates + ReplicatesPerBlock - 1) / ReplicatesPerBlock;
	// printf("Launching kernel with %i blocks of %i threads\n", blocksPerGrid, threadsPerBlock);
	AnnealingCoreGpu<Annealable><<<blocksPerGrid, threadsPerBlock>>>(
				task,
				WrkStates,         BestStates,
				WrkEnergies,       BestEnergies,
				ActArb,
				PwlTemp,           nOptions, nActions,
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
	ActArb.CopyDeviceToHost();
}

DispatchFunc* GpuTspDispatch = GpuDispatch<TspAnnealable>;
DispatchFunc* GpuIsingDispatch = GpuDispatch<IsingAnnealable>;
DispatchFunc* GpuPottsJitDispatch = GpuDispatch<PottsJitAnnealable>;
DispatchFunc* GpuPottsPrecomputeDispatch = GpuDispatch<PottsPrecomputeAnnealable>;
DispatchFunc* GpuPottsPrecomputePEDispatch = GpuDispatch<PottsPrecomputePEAnnealable>;