#include "Annealables.h"

//=====================================================================================constructor methods
__h__ IsingPrecomputeAnnealable::IsingPrecomputeAnnealable(PyObject* task, int nReplicates, bool USE_GPU){
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

	NHPP_potentials = NumCuda<double>(nReplicates, NumActions);
	UpdateIndices = NumCuda<int>(NumActions, kmap.dims[1]+1);

	//pre-build update indices.  Mostly copying over from kmap
	for (int i = 0; i < nPartitions; i++){
		UpdateIndices(i, 0) = kmap(i,0,0);
    	for (int c = 1; c < kmap(i,0,0); c++){
    		UpdateIndices(i, c) = kmap(i,c,2);
    	}
    	UpdateIndices(i, kmap(i,0,0)) = i; //becuase self always changes too
    }

    // printf("Prebuilt update indices successfully\n");

	if (USE_GPU) dispatch = GpuIsingPrecomputeDispatch;
    else         dispatch = CpuIsingPrecomputeDispatch;
}

__h__ __d__ float IsingPrecomputeAnnealable::EnergyOfState(int* state){
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

__h__ __d__ void IsingPrecomputeAnnealable::BeginEpoch(int iter){

	current_e = EnergyOfState(MiWrk);
    lowest_e = EnergyOfState(MiBest);

    //initialize NHPP potentials.
    for (int i = 0; i < nPartitions; i++){
    	NHPP_potentials(Rep, i) = biases(i, 1) - biases(i, 0);
    	for (int c = 1; c < kmap(i,0,0); c++){
    		int j = kmap(i,c,2); //index of the connected Potts node
    		float w = kmap(i,c,1); //scalar multiplier.  For Ising, this is all we need to know about the coupling.
    		NHPP_potentials(Rep, i) += w*MiWrk[j];
    	}
    }
}


__h__ __d__ void IsingPrecomputeAnnealable::FinishEpoch(){

}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__h__ __d__ float IsingPrecomputeAnnealable::GetActionDE(int action_num){
	return (1-2*MiWrk[action_num])*NHPP_potentials(Rep, action_num);
}

//changes internal state to reflect the annealing step that was taken
__h__ __d__ void IsingPrecomputeAnnealable::TakeAction_tic(int action_num){
	Mi_proposed_holder = 1 - MiWrk[action_num];
	//in order to ensure cooperative multi-threading correctness,
	//The proposed flip is recorded here, so that all threads can record the correct target state
	//and not get confused if another thread updates MiWrk in TakeAction_toc
	//before this thread reads the pre-exisiting state in TakeAction_toc

    current_e += GetActionDE(action_num);
}

__h__ __d__ void IsingPrecomputeAnnealable::TakeAction_toc(int action_num){
	if (action_num < 0) return;
	
    MiWrk[action_num] = Mi_proposed_holder; //needs to happen after all the GetActionDE calls
        
    if (current_e < lowest_e){
        lowest_e = current_e;
        for (int i = 0; i < nPartitions; i++) MiBest[i] = MiWrk[i];
    }

	//update NHPP_potentials:
	for (int c = 1; c < kmap(action_num,0,0); c++){
		int j = kmap(action_num,c,2); //index of the connected Potts node
		float w = kmap(action_num,c,1); //scalar multiplier.  For Ising, this is all we need to know about the coupling.
		if (Mi_proposed_holder == 1)
			NHPP_potentials(Rep, j) += w;
		else
			NHPP_potentials(Rep, j) -= w;
	}

	nRecentUpdates = UpdateIndices(action_num, 0); //first value holds the count
	recentUpdates = &UpdateIndices(action_num, 1); //second value onwards are the connections

}