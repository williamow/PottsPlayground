#include "Annealables.h"


//=====================================================================================constructor methods
PottsJitAnnealable::PottsJitAnnealable(PyObject* task, bool USE_GPU){

	// NumCuda<int> &kmap_,
	// 				  NumCuda<int> &qSizes_, 
	// 				  NumCuda<int> &partitions_,
	// 				  NumCuda<int> &partition_states_,
	// 				  NumCuda<float> &kernels_){
	kmap = 				NumCuda<int>(task, "kmap", 2, false, USE_GPU);
	qSizes = 			NumCuda<int>(task, "qSizes", 1, false, USE_GPU);
	partitions = 		NumCuda<int>(task, "Partitions", 1, false, USE_GPU);
	partition_states = 	NumCuda<int>(task, "Partition_states", 1, false, USE_GPU);
	kernels = 			NumCuda<float>(task, "kernels", 3, false, USE_GPU);
	biases = 			NumCuda<float>(task, "biases", 2, false, USE_GPU);

	nPartitions = qSizes.dims[0];
	nNHPPs = qSizes.sum();
	NumActions = nNHPPs;

	if (USE_GPU) dispatch = GpuDispatch<PottsJitAnnealable>;
    else         dispatch = CpuDispatch<PottsJitAnnealable>;

}

__host__ __device__ void PottsJitAnnealable::SetIdentity(int thread_num, int total_threads, int replicate_){
	Thrd = thread_num;
	nThrds = total_threads;
	Rep = replicate_;
}

__host__ __device__ void PottsJitAnnealable::SetState(int* &WrkState_, int* &BestState_, bool initialize){
	MiBest = BestState_;
	MiWrk = WrkState_;

	// std::mt19937 RngGenerator(0);
    // std::uniform_real_distribution<float> uniform_distribution(0.0,1.0);
    // #define RngUniform() uniform_distribution(RngGenerator)
    // #define RngInteger() RngGenerator()

	if (initialize){
		for (int partition = 0; partition < nPartitions; partition++){
            MiWrk[partition] = partition%qSizes(partition); //a really shitty pseudorandom initialization
            MiBest[partition] = MiWrk[partition];
        }
	}

	//intialize total energies to reflect the states that were just passed
	current_e = 0;
    lowest_e = 0;
    for (int i = 0; i < nPartitions; i++){
    	current_e += 2*biases(i, MiWrk[i]);
    	lowest_e += 2*biases(i, MiBest[i]);
        for (int j = 0; j < nPartitions; j++){
            current_e += kernels(kmap(i,j), MiWrk[i], MiWrk[j]);//[   qMax * qMax *kmap(i, j)  +  MiWrk[i]*qMax  +  MiWrk[j]  ];
            lowest_e += kernels(kmap(i,j), MiBest[i], MiBest[j]);
            //weights contribute to the active energy when both sides of the weights are selected.
        }
    }
    current_e = current_e / 2;
    lowest_e = lowest_e / 2;
}


__host__ __device__ void PottsJitAnnealable::FinalizeState(){
	
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
    for (int i = 0; i < nPartitions; i++) dE += kernels(kmap(action_partition, i), action_Mi, MiWrk[i]);
    //splitting these into different for loops might improve memory access times
    for (int i = 0; i < nPartitions; i++) dE -= kernels(kmap(action_partition, i), Mi_current, MiWrk[i]);
    
    last_action = action_num;
	last_dE = dE;
    return dE;
} 

//the potential energy; only really makes sense for Potts models.  Important only for my particular research purposes.
__host__ __device__ float PottsJitAnnealable::GetActionPE(int action_num){
	int action_partition = partitions(action_num);
	int action_Mi = partition_states(action_num);
	float E = 0;

    for (int i = 0; i < nPartitions; i++) E += kernels(kmap(action_partition, i), action_Mi, MiWrk[i]);

    return E;
} 

//changes internal state to reflect the annealing step that was taken
__host__ __device__ void PottsJitAnnealable::TakeAction_tic(int action_num){
	int action_partition = partitions(action_num);
	int action_Mi = partition_states(action_num);

    // if (action_num == last_action) current_e += last_dE;
    // else 
    current_e += GetActionDE(action_num); //super inneficient, fix! pass dE into here somehow...

    // MiWrk[action_partition] = action_Mi; //needs to happen after all the GetActionDE calls
        
    // if (current_e < lowest_e){
    //     lowest_e = current_e;
    //     for (int i = 0; i < nPartitions; i++) MiBest[i] = MiWrk[i];
    // }
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