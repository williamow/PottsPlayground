#include <stdio.h>
#include <random>
#include <thread>

#include "NumCuda.h"
#include "PieceWiseLinear.h"
#include "Annealables.h"

#include "Cores.h"

template <typename Annealable>  void AnnealingCoreCpu(
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

DispatchFunc* CpuTspDispatch = CpuDispatch<TspAnnealable>;
DispatchFunc* CpuIsingDispatch = CpuDispatch<IsingAnnealable>;
DispatchFunc* CpuPottsJitDispatch = CpuDispatch<PottsJitAnnealable>;
DispatchFunc* CpuPottsPrecomputeDispatch = CpuDispatch<PottsPrecomputeAnnealable>;
DispatchFunc* CpuPottsPrecomputePEDispatch = CpuDispatch<PottsPrecomputePEAnnealable>;