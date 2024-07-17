#include "Annealables.h"

//=====================================================================================constructor methods
__host__ PottsJitAnnealable::PottsJitAnnealable(PyObject* task, bool USE_GPU){

	
	qSizes = 			NumCuda<int>(task, "qSizes", 1, false, USE_GPU);
	partitions = 		NumCuda<int>(task, "Partitions", 1, false, USE_GPU);
	partition_states = 	NumCuda<int>(task, "Partition_states", 1, false, USE_GPU);
	kmap = 				NumCuda<float>(task, "kmap_sparse", 3, false, USE_GPU);
	kernels = 			NumCuda<float>(task, "kernels", 3, false, USE_GPU);
	biases = 			NumCuda<float>(task, "biases", 2, false, USE_GPU);

	nPartitions = qSizes.dims[0];
	nNHPPs = qSizes.sum();
	NumActions = nNHPPs;

	if (USE_GPU) dispatch = GpuDispatch<PottsJitAnnealable>;
    else         dispatch = CpuDispatch<PottsJitAnnealable>;
}

__host__ __device__ float PottsJitAnnealable::EnergyOfState(int* state){
	float e = 0;
    for (int i = 0; i < nPartitions; i++){
    	e += 2*biases(i, state[i]);
    	for (int c = 1; c < kmap(i,0,0); c++){
    		int j = kmap(i,c,2); //index of the connected Potts node
    		float w = kmap(i,c,1); //scalar multiplier
    		int k = kmap(i,c,0); //which kernel
    		e += w*kernels(k, state[i], state[j]);
    	}
    }
    return e/2;
}

__host__ __device__ void PottsJitAnnealable::BeginEpoch(int iter){

	if (iter==0){
		for (int partition = 0; partition < nPartitions; partition++){
            MiWrk[partition] = partition%qSizes(partition); //a really shitty pseudorandom initialization
            MiBest[partition] = MiWrk[partition];
        }
	}
	//intialize total energies to reflect the states that were just passed
	current_e = EnergyOfState(MiWrk);
    lowest_e = EnergyOfState(MiBest);
}


__host__ __device__ void PottsJitAnnealable::FinishEpoch(){
	// float eps = 1e-5;
	// float e = EnergyOfState(MiWrk);
	// if (current_e + eps < e || current_e - eps > e)
		// printf("Working Energy tracking error: actual=%.5f, tracked=%.5f\n", e, current_e); 
	
}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__host__ __device__ float PottsJitAnnealable::GetActionDE(int action_num){
	int action_partition = partitions(action_num);
	int action_Mi = partition_states(action_num);
	int Mi_current = MiWrk[action_partition];
	float dE = biases(action_partition, action_Mi) - biases(action_partition, Mi_current);
    //compute how much the energy would change.
    //assumes no weights between Mi_proposed and Mi_current, otherwise this calculation would be incorrect.

	for (int c = 1; c < kmap(action_partition,0,0); c++){
		int j = kmap(action_partition,c,2); //index of the connected Potts node
		float w = kmap(action_partition,c,1); //scalar multiplier
		int k = kmap(action_partition,c,0); //which kernel
		dE += w*kernels(k, action_Mi, MiWrk[j]);
		dE -= w*kernels(k, Mi_current, MiWrk[j]);
	}

    return dE;
}

//changes internal state to reflect the annealing step that was taken
__host__ __device__ void PottsJitAnnealable::TakeAction_tic(int action_num){
	int action_partition = partitions(action_num);
	int action_Mi = partition_states(action_num);

    current_e += GetActionDE(action_num);
}

__host__ __device__ void PottsJitAnnealable::TakeAction_toc(int action_num){
	if (action_num < 0) return;
	int action_partition = partitions(action_num);
	int action_Mi = partition_states(action_num);
	
    MiWrk[action_partition] = action_Mi; //needs to happen after all the GetActionDE calls
        
    if (current_e < lowest_e){
        lowest_e = current_e;
        for (int i = 0; i < nPartitions; i++) MiBest[i] = MiWrk[i];
    }
}