#include <stdio.h>
#include <random>
#include <cuda_runtime.h>
#include <typeinfo>
#include "Annealables.h"



	//=====================================================================================constructor methods
PottsPrecomputeAnnealable::PottsPrecomputeAnnealable(PyObject *task, int nReplicates, bool USE_GPU){

	qSizes = 			NumCuda<int>(task, "qSizes", 1, false, USE_GPU);
	qCumulative = 		NumCuda<int>(task, "qCumulative", 1, false, USE_GPU);
	partitions = 		NumCuda<int>(task, "Partitions", 1, false, USE_GPU);
	partition_states = 	NumCuda<int>(task, "Partition_states", 1, false, USE_GPU);
	kmap = 				NumCuda<float>(task, "kmap_sparse", 3, false, USE_GPU);
	dense_kernels = 	NumCuda<float>(task, "kernels", 3, false, USE_GPU);
	biases = 			NumCuda<float>(task, "biases", 2, false, USE_GPU);

	nPartitions = qSizes.dims[0];
	nNHPPs = qSizes.sum();
	NumActions = nNHPPs;
	NHPP_potentials = NumCuda<double>(nReplicates, nNHPPs);

	int nKernels = dense_kernels.dims[0];
	int qMax = dense_kernels.dims[1];
	sparse_kernels = NumCuda<float>(nKernels, qMax, qMax+1, 2); //there is likely at least one dense kernel, so might as well allocate the full qMax+1

	// printf("mark 1\n");
	//convert dense kernels to sparse kernels:
	for (int k = 0; k < nKernels; k++){
		for (int i = 0; i < qMax; i++){
			int c = 1;
			for (int j = 0; j<qMax; j++){
				if (dense_kernels(k, i, j) != 0){
					sparse_kernels(k,i,c,0) = j;
					sparse_kernels(k,i,c,1) = dense_kernels(k, i, j);
					c++;
					// printf("K=%i, i=%i, c=%i, j=%i, w=%f\n", k, i, c, j, dense_kernels(k, i, j));
				}
			sparse_kernels(k, i, 0, 0) = c;
			}
		}
	}
	sparse_kernels.CopyHostToDevice();

	if (USE_GPU) dispatch = GpuDispatch<PottsPrecomputeAnnealable>;
    else         dispatch = CpuDispatch<PottsPrecomputeAnnealable>;
}

__host__ __device__ float PottsPrecomputeAnnealable::EnergyOfState(int* state){
	float e = 0;
    for (int i = 0; i < nPartitions; i++){
    	e += 2*biases(i, state[i]);
    	for (int c = 1; c < kmap(i,0,0); c++){
    		int j = kmap(i,c,2); //index of the connected Potts node
    		float w = kmap(i,c,1); //scalar multiplier
    		int k = kmap(i,c,0); //which kernel
    		e += w*dense_kernels(k, state[i], state[j]);
    		// for (int d = 1; d < sparse_kernels(k, state[i], 0, 0); d++){
    			// if (sparse_kernels(k, state[i], d, 0) == state[j]) e += w*sparse_kernels(k, state[i], d, 1);
    		// }
    	}
    }
    return e/2;
}

__host__ __device__ void PottsPrecomputeAnnealable::BeginEpoch(int iter){
	// printf("mark 2\n");
	if (iter==0){
		for (int partition = 0; partition < nPartitions; partition++){
			// printf("qSizes[%i]=%i\n", partition, qSizes(partition));
            MiWrk[partition] = partition%qSizes(partition); //"pseudorandom" initialization that is consistent between threads working together
            MiBest[partition] = MiWrk[partition];
        }
	}
	// printf("mark 3\n");
	//intialize total energies to reflect the states that were just passed
	current_e = EnergyOfState(MiWrk);
	lowest_e = EnergyOfState(MiBest);

	// printf("mark 4\n");
    //need to make sure that the potentials this thread sets to zero are the same that it calculates the starting energy for
    for (int i=Thrd; i<nPartitions; i+=nThrds){
		int NHPP = qCumulative(i)-qSizes(i);
		for (int q=0; q<qSizes(i); q++){
			NHPP_potentials(Rep, NHPP+q) = biases(i,q);
		}
	}
	// printf("mark 5\n");
    for (int i=Thrd; i<nPartitions; i+=nThrds){
    	for (int c = 1; c < kmap(i, 0, 0); c++){
    		int j = kmap(i,c,2); //index of the connected Potts node
    		if (j==i) printf("Row %i has a connection to Column %i. This is unsupported\n", i, j);
    		float w = kmap(i,c,1); //scalar multiplier
    		int k = kmap(i,c,0); //which kernel
    		int NHPP = qCumulative(i)-qSizes(i);
    		for (int q = 0; q < qSizes(i); q++){
    			NHPP_potentials(Rep, NHPP+q) += w*dense_kernels(k, q, MiWrk[j]);
    		}
    		// for (int d = 1; d < sparse_kernels(k, MiWrk[j], 0, 0); d++){
    			// int q = sparse_kernels(k, MiWrk[j], d, 0);
    			// NHPP_potentials(Rep, NHPP+q) += w*sparse_kernels(k, MiWrk[j], d, 1);
    		// }
    	}
    }
}

__host__ __device__ void PottsPrecomputeAnnealable::FinishEpoch(){
	float eps = 1;
	float e = EnergyOfState(MiWrk);
	if (current_e + eps < e || current_e - eps > e)
		printf("Working Energy tracking error: actual=%.5f, tracked=%.5f\n", e, current_e); 
}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__host__ __device__ float PottsPrecomputeAnnealable::GetActionDE(int action_num){
	int action_partition = partitions(action_num);
	int new_Mi = partition_states(action_num);
	int old_Mi = MiWrk[action_partition];
	int old_NHPP = qCumulative(action_partition)-qSizes(action_partition) + old_Mi;
	// printf("Action: (M=%i, q=%i, pE=%.2f) -> (M=%i, q=%i, pE=%.2f)\n",
		// action_partition, old_Mi, NHPP_potentials(Rep, old_NHPP), action_partition, new_Mi, NHPP_potentials(Rep, action_num));
    return NHPP_potentials(Rep, action_num) - NHPP_potentials(Rep, old_NHPP);
} 

//changes internal state to reflect the annealing step that was taken
__host__ __device__ void PottsPrecomputeAnnealable::TakeAction_tic(int action_num){
	float dE = GetActionDE(action_num);
	// printf("taking action %i with dE %.5f\n", action_num, dE);
	current_e += dE;
}

__host__ __device__ void PottsPrecomputeAnnealable::TakeAction_toc(int action_num){
	if (action_num < 0) return;

	int j = partitions(action_num);
	int new_Mi = partition_states(action_num);
	int old_Mi = MiWrk[j];
	int old_NHPP = qCumulative(j)-qSizes(j) + old_Mi;
	MiWrk[j] = new_Mi;
        
    if (current_e < lowest_e){
        lowest_e = current_e;
        for (int i = Thrd; i < nPartitions; i+=nThrds) MiBest[i] = MiWrk[i];
    }

	//adjust all of the potential energies to reflect the new working state:
    for (int c = 1; c < kmap(j, 0, 0); c++){
		int i = kmap(j,c,2); //index of the connected Potts node
		if (i%nThrds != Thrd) continue; //each worker thread is only allowed to work on a particular subset of the partitions
		float w = kmap(j,c,1); //scalar multiplier
		int k = kmap(j,c,0); //which kernel
		int NHPP = qCumulative(i)-qSizes(i);
		for (int d = 1; d < sparse_kernels(k, old_Mi, 0, 0); d++){
			int q = sparse_kernels(k, old_Mi, d, 0);
			float ww = w*sparse_kernels(k, old_Mi, d, 1);
			NHPP_potentials(Rep, NHPP+q) -= ww;
		}
		for (int d = 1; d < sparse_kernels(k, new_Mi, 0, 0); d++){
			int q = sparse_kernels(k, new_Mi, d, 0);
			float ww = w*sparse_kernels(k, new_Mi, d, 1);
			NHPP_potentials(Rep, NHPP+q) += ww;
		}
    }


}