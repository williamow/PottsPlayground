#include <stdio.h>
#include <random>
#include <thread>

#include "NumCuda.h"
#include "PieceWiseLinear.h"
#include "Annealables.h"

#include "Cores.h"

template <typename Annealable> void OptionsActionsCpu(
	Annealable &&task,
	DispatchArgs &&DA,
	int replicate_index,
	volatile int *TaskDone){

	std::mt19937 RngGenerator(replicate_index + DA.MinIter*1000);
	std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);

	//create local references to the state vectors used in this particular thread:
	int *MiWrk = &DA.WrkStates(replicate_index, 0);
	int *MiBest = &DA.BestStates(replicate_index, 0);

	task.SetIdentity(0, 1, replicate_index, MiWrk, MiBest);
	task.BeginEpoch(DA.MinIter);

	bool FullParallel = (DA.nOptions >= task.NumActions);

	float init_priority = 0.7;
	if (!strcmp(DA.pMode, "Force"))
		init_priority = 1e9;

	for (long iter = DA.MinIter; iter < DA.MaxIter; iter+=DA.nActions){
		if (iter%1000000 == 0){
			//re-init task, to eliminate floating point errors that sometimes accumulate,
			//since the total energy of the system is updated incrimentally over and over again
			task.FinishEpoch();
			task.BeginEpoch(iter);
		}


		if (*TaskDone > 0) break;
		
		float T = DA.PwlTemp.interp(iter);

		//initializing values for action selection arbitration
		for (int act = 0; act<DA.nActions; act++)
			DA.ActArb(replicate_index, act, 0) = init_priority;//-log(uniform_distribution(RngGenerator));
			//null priorities are initialized here since it uses device-dependent rng code

		//using CTMC math and the options continuum:
		
		ActionArbitrator Arb(&DA.ActArb, replicate_index);

		for (int i=0; i<DA.nOptions; i++){
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
		if (iter + DA.nActions >= DA.MaxIter) break;

		for (int act = 0; act<DA.nActions; act++){
			int action = DA.ActArb(replicate_index, act, 1);
			// printf("Taking action %i with priority %.3f\n", action, ActArb(replicate_index, act, 0));
			if (action >= 0) {
				task.TakeAction_tic(action);
				DA.RealFlips(replicate_index)++;
			}
			task.TakeAction_toc(action);
		}
		task.check_energy();

		if (task.lowest_e < DA.e_th) *TaskDone = 1; 
	}

	task.FinishEpoch();
	DA.WrkEnergies(replicate_index) = task.current_e;
	DA.BestEnergies(replicate_index) = task.lowest_e;
}

// template <typename Annealable> void ParallelTrials(
// 	Annealable && task,
// 	DispatchArgs &&DA,
// 	int replicate_index,
// 	volatile int *TaskDone){

// 	std::mt19937 RngGenerator(replicate_index + DA.MinIter*1000);
// 	std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);

// 	int *MiWrk = &DA.WrkStates(replicate_index, 0);
// 	int * MiBest = &DA.BestStates(replicate_index, 0);

// 	task.SetIdentity(0, 1, replicate_index, MiWrk, MiBest);
// 	task.BeginEpoch(DA.MinIter);

// 	NumCuda<int> flip_flags(DA.nOptions);
// 	float offst = 0.0001;

// 	bool FullParallel = (DA.nOptions >= task.NumActions);

// 	int iter_inc = 1;
// 	if (DA.ParallelUpdates) iter_inc = DA.nActions;
// 	for (long iter = DA.MinIter; iter < DA.MaxIter; iter+=iter_inc){

// 		if (*TaskDone > 0) break;
		
// 		float T = DA.PwlTemp.interp(iter);

// 		int numFlips = 0;
// 		for (int i = 0; i < DA.nOptions; i++){
// 			int trial = FullParallel ? i : RngGenerator()%task.NumActions;
// 			if (trial >= task.NumActions) break;
// 			float dE = task.GetActionDE(trial);
// 			float prob = offst*exp(-dE/T);
// 			if (!strcmp(DA.pMode, "NHPP")) prob = 1 - exp(-prob); //poisson process probability
// 			if (prob > uniform_distribution(RngGenerator)){
// 				flip_flags(numFlips) = trial;
// 				numFlips++;
// 				printf("%.2f\n",prob);
// 			}
// 		}

// 		//feedback loop to adjust DE offset, so that numFlips matches the target.
// 		//uses cumulative target/actual so that it ajusts faster at first and slower towards the end
// 		// offst = offst*((iter+1)/(DA.RealFlips(replicate_index)+1+numFlips));
// 		printf("%i iters, %i flips, offst=%.5f\n",iter,numFlips,offst);
// 		offst = offst*(0.0 + 1.0*(DA.nActions+1)/(numFlips+1));
		//feedback loop to adjust DE offset, so that numFlips matches the target.
		//uses cumulative target/actual so that it ajusts faster at first and slower towards the end
		// offst = offst*((iter+1)/(DA.RealFlips(replicate_index)+1+numFlips));
		// printf("%li iters, %i flips, offst=%.5f\n",iter,numFlips,offst);
		// offst = offst*(0.0 + 1.0*(DA.nActions+1)/(numFlips+1));

// 		if (numFlips == 0) continue;

// 		if (!DA.ParallelUpdates){ //if serial only updates, select only one action to take
// 			int action = floor(uniform_distribution(RngGenerator)*numFlips);
// 			action = flip_flags(action);
// 			flip_flags(0) = action;
// 			numFlips = 1;
// 		}

// 		for (int i = 0; i < numFlips; i++){
// 			int action = flip_flags(i);
// 			task.TakeAction_tic(action);
// 			DA.RealFlips(replicate_index)++;
// 			task.TakeAction_toc(action);
// 		}
// 		task.check_energy();

// 	}

// 	task.FinishEpoch();
// 	DA.WrkEnergies(replicate_index) = task.current_e;
// 	DA.BestEnergies(replicate_index) = task.lowest_e;
// }

template <typename Annealable> void KmcCpu(
	//kinetic monte carlo, previously called BirdsEye.
	//Use logarithmic decision tree to pick out the new update at each step.
	//Only recalculate exponentials, etc. when Delta E changes;
	//Unaltered options may become out of date, since temperature changes at each
	//cycle will not be reflected in all NHPP rate updates.
	//Also, guess I am redesigning the models part to no longer distinguish PottsJit and PottsPrecompute,
	//moving the precompute functionality into this code bit?
	//hmm no.  Let's keep the precompute where it is, but have the precompute step return a list of modified energies!
	//yeah! actually that's a superb idea, I think. Just fantanstic.

	Annealable &&task,
	DispatchArgs &&DA,
	int replicate_index,
	volatile int *TaskDone){

	std::mt19937 RngGenerator(replicate_index + DA.MinIter*1000);
	std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);

	//create local references to the state vectors used in this particular thread:
	int *MiWrk = &DA.WrkStates(replicate_index, 0);
	int *MiBest = &DA.BestStates(replicate_index, 0);

	task.SetIdentity(0, 1, replicate_index, MiWrk, MiBest);
	task.BeginEpoch(DA.MinIter);

	//initialize energy-mapping decision tree thing:
	int tree_levels = 1+ceil(log2(task.NumActions));
	NumCuda<float> NHPP_rates(tree_levels, task.NumActions);
	for (int i=0; i<tree_levels; i++)
		for (int j=0; j<task.NumActions; j++)
			NHPP_rates(i,j) = 0;

	int nChanges = task.NumActions; 

	for (long iter = DA.MinIter; iter < DA.MaxIter; iter++){

		if (*TaskDone > 0) break;
		
		float T = DA.PwlTemp.interp(iter);
		// printf("nChanges: %i\n",nChanges);
		for (int i = 0; i<nChanges; i++){
			int changed_action = (nChanges == task.NumActions) ? i : task.recentUpdates[i];
			float dE = task.GetActionDE(changed_action);
			float old_rate = NHPP_rates(0,changed_action);
			NHPP_rates(0,changed_action) = exp(-dE/T);
			// printf("Action: %i, rate: %f\n", changed_action, NHPP_rates(0,changed_action));
			float rate_change = NHPP_rates(0,changed_action) - old_rate;
			//now, propagate the change up the tree:
			for (int lvl = 1; lvl < tree_levels; lvl++){
				int lvl_pos = changed_action >> lvl;
				NHPP_rates(lvl, lvl_pos) = NHPP_rates(lvl, lvl_pos) + rate_change;
			}
		} 

		//now, select one of the NHPPs as having fired first:
		int action = 0;
		float r = uniform_distribution(RngGenerator)*NHPP_rates(tree_levels-1, 0);
		// printf("z=%.2f\n", NHPP_rates(tree_levels-1, 0));
		for (int lvl = tree_levels-2; lvl >= 0; lvl = lvl - 1){
			action = action << 1;
			// printf("lvl: %i, Action: %i, r: %f, NHPPs: %f,%f\n", lvl, action, r, NHPP_rates(lvl, action), NHPP_rates(lvl, action+1));
			if (r > NHPP_rates(lvl, action)){
				r = r - NHPP_rates(lvl, action);
				action++;
			}
		} 
		// printf("Final action: %i\n", action);
		if (action >= task.NumActions){
			//implies that a lot of floating-point errors have accumulated,
			//so the tree should be re-built. 
			//Set everything in the tree back to zero, and set nChanges to NumActions:
			for (int i=0; i<tree_levels; i++)
				for (int j=0; j<task.NumActions; j++)
					NHPP_rates(i,j) = 0;

			nChanges = task.NumActions;
			continue; //skip taking an action
		} 
		task.TakeAction_tic(action);
		DA.RealFlips(replicate_index)++;
		task.TakeAction_toc(action);

		task.check_energy();

		nChanges = task.nRecentUpdates;
		if (nChanges < 0) nChanges = task.NumActions; 

		if (task.lowest_e < DA.e_th) *TaskDone = 1; 
	}

	task.FinishEpoch();
	DA.WrkEnergies(replicate_index) = task.current_e;
	DA.BestEnergies(replicate_index) = task.lowest_e;
}

template <typename Annealable> void MetropolisHastingsCpu(
	Annealable &&task,
	DispatchArgs &&DA,
	int replicate_index,
	volatile int *TaskDone){

	std::mt19937 RngGenerator(replicate_index + DA.MinIter*1000);
	std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);

	//create local references to the state vectors used in this particular thread:
	int *MiWrk = &DA.WrkStates(replicate_index, 0);
	int *MiBest = &DA.BestStates(replicate_index, 0);

	task.SetIdentity(0, 1, replicate_index, MiWrk, MiBest);
	task.BeginEpoch(DA.MinIter);

	for (long iter = DA.MinIter; iter < DA.MaxIter; iter++){
		if (iter%1000000 == 0){
			//re-init task, to eliminate floating point errors that sometimes accumulate,
			//since the total energy of the system is updated incrimentally over and over again
			task.FinishEpoch();
			task.BeginEpoch(iter);
		} 

		if (*TaskDone > 0) break;
		
		float T = DA.PwlTemp.interp(iter);

		int action = RngGenerator()%task.NumActions;
		float possible_dE = task.GetActionDE(action);
		float pr_action = exp(-possible_dE/T);
		if (pr_action > uniform_distribution(RngGenerator)){
			task.TakeAction_tic(action);
			DA.RealFlips(replicate_index)++;
			task.TakeAction_toc(action);
			task.check_energy();	
		} 

		if (task.lowest_e < DA.e_th) *TaskDone = 1; 
	}

	task.FinishEpoch();
	DA.WrkEnergies(replicate_index) = task.current_e;
	DA.BestEnergies(replicate_index) = task.lowest_e;
}


//core annealing is further wrapped in template dispatch functions, 
//which supports sudo-multi-dispatch function calling in the main routine.
//The main routine gets pointers to these functions,
//which all have the same call signature even though they are individually compiled for each type of annealable object.
//Since this requires passing the task via pointer, we can't directly dispatch the kernel for either CPU or GPU,
//since then all threads would have the same task copy/the GPU would have a pointer to host memory.
 
template <typename Annealable> void CpuDispatch(void* void_task, DispatchArgs &DA){

	Annealable task = *(Annealable*)void_task; //converts the void pointer to an Annealable pointer, dereferences it, and makes a copy

	// int nWorkers = ActArb.dims[1]; //for CPU dispatch, nWorkers instead specifies how many CPU threads can run in parallel
	int nReplicates = DA.ActArb.dims[0];

	int nWorkers = DA.nWorkers; //copy, since the value might be modified:
	if (nWorkers < 0) nWorkers = std::thread::hardware_concurrency()/(-nWorkers);
	//negative values allow automatic full use of processor.
	//divides by negative value, i.e. -1 uses all threads, -2 uses half, etc.
	//Sorry for the over-utilization of a single parameter!

	if (nWorkers < 1) nWorkers = 1;


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
			if (!strcmp(DA.algo, "OptionsActions") || !strcmp(DA.algo, "ParallelTrials"))
				workers.push_back(std::thread(OptionsActionsCpu<Annealable>, task, DA, thread_num, DA.GlobalHalt.hd));
			
			else if (!strcmp(DA.algo, "BirdsEye") || !strcmp(DA.algo, "KMC")) //KMC is new term
				workers.push_back(std::thread(KmcCpu<Annealable>, task, DA, thread_num, DA.GlobalHalt.hd));
			
			else
				workers.push_back(std::thread(MetropolisHastingsCpu<Annealable>, task, DA, thread_num, DA.GlobalHalt.hd));

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

DispatchFunc* CpuSwapDispatch = CpuDispatch<SwapAnnealable>;
DispatchFunc* CpuIsingDispatch = CpuDispatch<IsingAnnealable>;
DispatchFunc* CpuIsingPrecomputeDispatch = CpuDispatch<IsingPrecomputeAnnealable>;
DispatchFunc* CpuPottsJitDispatch = CpuDispatch<PottsJitAnnealable>;
DispatchFunc* CpuPottsPrecomputeDispatch = CpuDispatch<PottsPrecomputeAnnealable>;
DispatchFunc* CpuPottsPrecomputePEDispatch = CpuDispatch<PottsPrecomputePEAnnealable>;