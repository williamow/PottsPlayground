#include <stdio.h>
#include <random>
#include <thread>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "NumCuda.hpp"
#include "PieceWiseLinear.cpp"

//little class for keeping track of which actions should be taken and which should be discarded.
class ActionArbitrator{

public:
NumCuda<float> *ActArb;
int nActions;
int worstActionOfTheBest;
float worstPriorityOfTheBest;
int replicate_index;

__host__ __device__ inline ActionArbitrator(NumCuda<float> * ActArb_, int replicate_index_){
	replicate_index = replicate_index_;
	ActArb = ActArb_;
	nActions = ActArb->dims[1];

	for (int act = 0; act<nActions; act++){
		//initialize each to the default Null action, with dE=0.
		(*ActArb)(replicate_index, act, 1) = -1; 
		// printf("Arb slot %i: Priority = %.3f, action = %i\n", act, (*ActArb)(replicate_index, act, 0), int((*ActArb)(replicate_index, act, 1)));
	}

	UpdateWorst();
}

__host__ __device__ inline void UpdateWorst(){
	//find which option is now the worst, to be booted first the next time there is an update
	worstActionOfTheBest = 0;
	worstPriorityOfTheBest = (*ActArb)(replicate_index, 0, 0);

	for (int act = 0; act<nActions; act++){
		if ((*ActArb)(replicate_index, act, 0) > worstPriorityOfTheBest){
			worstPriorityOfTheBest = (*ActArb)(replicate_index, act, 0);
			worstActionOfTheBest = act;
		}
	}
}

__host__ __device__ inline void AddOption(int NewOption, float NewPriority){
	// printf("New Priority = %.3f, New action = %i\n", NewPriority, NewOption);
	if ((*ActArb)(replicate_index, worstActionOfTheBest, 0) != worstPriorityOfTheBest) UpdateWorst(); //if another GPU thread made a change
	if (NewPriority < worstPriorityOfTheBest){
		//replace old worst option with the new option
		(*ActArb)(replicate_index, worstActionOfTheBest, 0) = NewPriority;
		(*ActArb)(replicate_index, worstActionOfTheBest, 1) = NewOption;
		// printf("Changing Arb slot %i to Priority = %.3f, action = %i\n", worstActionOfTheBest, NewPriority, NewOption);

		UpdateWorst();
	}
}
};


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

template <typename Annealable> __host__  void AnnealingCoreCpu(
	Annealable &&task,
	NumCuda<int> &&working_states,
	NumCuda<int> &&best_states,
	NumCuda<float> &&working_energies,
	NumCuda<float> &&best_energies,
	NumCuda<float> &&ActArb, //for communication between cooperating threads; use however seen fit.
	NumCuda<int> &&RealFlips,
	PieceWiseLinear &&PwlTemp, //for calculating the annealing temperature to use.
	int nOptions,
	int nActions, //how many actions to actually take at each cycle
	long MinIter,
	long MaxIter,
	float e_th, //energy threshold to trigger premature termination
	int replicate_index,
	volatile int *TaskDone //bit that tells all processes to exit, set by a single process upon finding a 'solution'.
	//super important that this is declared volatile, so that it is always retreived from global memory, 
){

	std::mt19937 RngGenerator(replicate_index + MinIter*1000);
	std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);

	// int nThrds = 1;//ActArb.dims[1]; //threads per replicate
	int nReplicates = ActArb.dims[0];
	// int replicate_index = ThrdIndx;///nThrds;
	// int Thrd = 1;//ThrdIndx%nThrds; //thread index within each replicate
	if (replicate_index + 1 > nReplicates) return;

	//create local references to the state vectors used in this particular thread:
	int *MiWrk = &working_states(replicate_index, 0);
	int *MiBest = &best_states(replicate_index, 0);

	task.SetIdentity(0, 1, replicate_index, MiWrk, MiBest);
	task.BeginEpoch(MinIter);

	bool FullParallel = (nOptions >= task.NumActions);

	for (long iter = MinIter; iter < MaxIter; iter+=nActions){
		if (iter%1000000 == 0){
			//re-init task, to eliminate floating point errors that sometimes accumulate,
			//since the total energy of the system is updated incrimentally over and over again
			task.FinishEpoch();
			task.BeginEpoch(iter);
		}


		if (*TaskDone > 0) break;
		
		float T = PwlTemp.interp(iter);

		//initializing values for action selection arbitration
		for (int act = 0; act<nActions; act++)
			ActArb(replicate_index, act, 0) = -log(uniform_distribution(RngGenerator));
			//null priorities are initialized here since it uses device-dependent rng code

		ActionArbitrator Arb(&ActArb, replicate_index);

		for (int i=0; i<nOptions; i++){
			int NewOption = FullParallel ? i : RngGenerator()%task.NumActions;
			if (NewOption >= task.NumActions) break;
			float possible_dE = task.GetActionDE(NewOption);

			//switched signs in order to convert divisions to multiplications:
			float NhppRate = exp(possible_dE/T); //exp2(-possible_dE/T); 
			float NewPriority = -log(uniform_distribution(RngGenerator))*NhppRate; ///NhppRate;
			// printf("dE: %.2f\n", possible_dE);
			Arb.AddOption(NewOption, NewPriority); //if the new option has a higher priority, it will be added to ActArb memory
		}

		//skip the last set of actions, 
		//so the last state and its dwell time can be recorded together for technically correct sampling probabilities
		if (iter + nActions >= MaxIter) break;

		for (int act = 0; act<nActions; act++){
			int action = ActArb(replicate_index, act, 1);
			// printf("Taking action %i with priority %.3f\n", action, ActArb(replicate_index, act, 0));
			if (action >= 0) {
				task.TakeAction_tic(action);
				RealFlips(replicate_index)++;
			}
			task.TakeAction_toc(action);
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

//defines format of the dispatch functions, for creating function pointers more easily
using DispatchFunc = void(
		void*, //the task, which must be specified as a void pointer since the actual type changes but I want an unchanging function interface
		NumCuda<int> &, NumCuda<int> &, //working states, best states
		NumCuda<float> &, NumCuda<float> &, //working energies, best energies
		NumCuda<float> &, //communal space memory for interthread cooperation
		NumCuda<int> &, //For returning the total number of actual flips
		PieceWiseLinear, int, int, int, //annealing temperature specification, nOptions, nActions, nWorkers
		long, long, //min iter, max iter
		float, //termination threshold energy
		NumCuda<int> &//global halt
		);


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

// template <typename T> void testfunc(NumCuda<int> &&garbage){
	// printf("%i\n", garbage.dims[0]);
// }

template <typename Annealable> void CpuDispatch(
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
		NumCuda<int> &GlobalHalt
		){

	Annealable task = *(Annealable*)void_task; //converts the void pointer to an Annealable pointer, dereferences it, and makes a copy

	// int nWorkers = ActArb.dims[1]; //for CPU dispatch, nWorkers instead specifies how many CPU threads can run in parallel
	int nReplicates = ActArb.dims[0];

	

	// std::thread t1(testfunc<int>, WrkStates);

	// std::barrier<> * b; 
	//tried using barriers to make CPU thread worker groups. 
	//Unfortunately CPU Thread scheduling makes this very slow, so I've removed it.
	//Now different threads just compute their own replicates.


	//manage a vector of worker threads, never more than nWorkers active at the same time,
	//until nReplicates threads have been launched and completed
	std::vector<std::thread> workers;

	int thread_num = 0;
	while (thread_num < nReplicates){
		if (workers.size() < nWorkers){
			workers.push_back(std::thread(AnnealingCoreCpu<Annealable>,
					task,
					WrkStates,         BestStates,
					WrkEnergies,       BestEnergies,
					ActArb,
					RealFlips,
					PwlTemp,           
					nOptions, 	   nActions, //take all actions is now an int that says how many actions to take
					MinIter,           MaxIter,
					e_th,
					thread_num,
					// b,
					GlobalHalt.hd));
			thread_num++;
		}
		else {
			//wait for first current thread to finish, and delete it when done
			if (workers[0].joinable()) workers[0].join();
			workers.erase(workers.begin()); //if already done and not joinable, erase it also
		}
	}

	//wait for all remaining threads to finish
	for (std::thread &t: workers) {
	  if (t.joinable()) {
		t.join();
	  }
	}
}