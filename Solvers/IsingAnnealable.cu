#include "Annealables.h"

//=====================================================================================constructor methods
__host__ IsingAnnealable::IsingAnnealable(PyObject* task, bool USE_GPU){
	//accelerate ising-only computation by taking advantage of all the simplifications when q=2.
	//operates on the same problem format as the Potts solvers,
	//but assumes that there is a single kernel that meets requirements for Binary Quadratic Format,
	//so that the kernel can be discarded and only the kmap weight actually needs to be used.
	
	NumCuda<int> qSizes = 	NumCuda<int>(task, "qSizes", 1, false, USE_GPU);
	kmap = 					NumCuda<float>(task, "kmap_sparse", 3, false, USE_GPU);
	biases = 				NumCuda<float>(task, "biases", 2, false, USE_GPU);

	//temporarily load kernels, to make sure that requirements are met:
	// TODO 

	nPartitions = qSizes.dims[0];
	nNHPPs = qSizes.sum();
	NumActions = nPartitions; //each action is now just flipping a state, so there is only 1 per partition

	if (USE_GPU) dispatch = GpuDispatch<IsingAnnealable>;
    else         dispatch = CpuDispatch<IsingAnnealable>;
}

__host__ __device__ float IsingAnnealable::EnergyOfState(int* state){
	float e = 0;
    for (int i = 0; i < nPartitions; i++){
    	e += 2*biases(i, state[i]);
    	for (int c = 1; c < kmap(i,0,0); c++){
    		int j = kmap(i,c,2); //index of the connected Potts node
    		float w = kmap(i,c,1); //scalar multiplier.  For Ising, this is all we need to know about the coupling.
    		e += w*state[i]*state[j];
    	}
    }
    return e/2;
}

__host__ __device__ void IsingAnnealable::BeginEpoch(int iter){

	if (iter==0){
		for (int partition = 0; partition < nPartitions; partition++){
            MiWrk[partition] = partition%2; //a really shitty pseudorandom initialization
            MiBest[partition] = MiWrk[partition];
        }
	}
	//intialize total energies to reflect the states that were just passed
	current_e = EnergyOfState(MiWrk);
    lowest_e = EnergyOfState(MiBest);
}


__host__ __device__ void IsingAnnealable::FinishEpoch(){
	// float eps = 1e-5;
	// float e = EnergyOfState(MiWrk);
	// if (current_e + eps < e || current_e - eps > e)
		// printf("Working Energy tracking error: actual=%.5f, tracked=%.5f\n", e, current_e); 
	
}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__host__ __device__ float IsingAnnealable::GetActionDE(int action_num){
	int action_partition = action_num;
	int Mi_current = MiWrk[action_partition];
	int Mi_proposed = 1 - Mi_current; //since action is defined as a flip

	float dE = biases(action_partition, Mi_proposed) - biases(action_partition, Mi_current);
    //compute how much the energy would change.
    //assumes no weights between Mi_proposed and Mi_current, otherwise this calculation would be incorrect.

	for (int c = 1; c < kmap(action_partition,0,0); c++){
		int j = kmap(action_partition,c,2); //index of the connected Potts node
		float w = kmap(action_partition,c,1); //scalar multiplier
		dE += w*MiWrk[j]*(Mi_proposed-Mi_current);
		// dE += w*kernels(k, action_Mi, MiWrk[j]);
		// dE -= w*kernels(k, Mi_current, MiWrk[j]);
	}

    return dE;
}

//changes internal state to reflect the annealing step that was taken
__host__ __device__ void IsingAnnealable::TakeAction_tic(int action_num){
	Mi_proposed_holder = 1 - MiWrk[action_num];
	//in order to ensure cooperative multi-threading correctness,
	//The proposed flip is recorded here, so that all threads can record the correct target state
	//and not get confused if another thread updates MiWrk in TakeAction_toc 
	//before this thread reads the pre-exisiting state in TakeAction_toc

    current_e += GetActionDE(action_num);
}

__host__ __device__ void IsingAnnealable::TakeAction_toc(int action_num){
	if (action_num < 0) return;
	
    MiWrk[action_num] = Mi_proposed_holder; //needs to happen after all the GetActionDE calls
        
    if (current_e < lowest_e){
        lowest_e = current_e;
        for (int i = 0; i < nPartitions; i++) MiBest[i] = MiWrk[i];
    }
}