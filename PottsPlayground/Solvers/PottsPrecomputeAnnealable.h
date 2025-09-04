#include <stdio.h>
#include <random>
#include <typeinfo>
#include "Annealables.h"



	//=====================================================================================constructor methods
PottsPrecomputeBase::PottsPrecomputeBase(PyObject *task, int nReplicates, bool USE_GPU){

	qSizes = 			NumCuda<int>(task, "qSizes", 1, false, USE_GPU);
	qCumulative = 		NumCuda<int>(task, "qCumulative", 1, false, USE_GPU);
	// partitions = 		NumCuda<int>(task, "Partitions", 1, true, USE_GPU);
	// partition_states = 	NumCuda<int>(task, "Partition_states", 1, false, USE_GPU);
	kmap = 				NumCuda<float>(task, "kmap_sparse", 3, false, USE_GPU);
	dense_kernels = 	NumCuda<float>(task, "kernels", 3, false, USE_GPU);
	biases = 			NumCuda<float>(task, "biases", 2, false, USE_GPU);

	nPartitions = qSizes.dims[0];
	nNHPPs = qSizes.sum();
	NumActions = nNHPPs-nPartitions;
	NHPP_potentials = NumCuda<double>(nReplicates, nNHPPs);
	UpdateTracker = NumCuda<int>(nReplicates, nNHPPs);

	//create partition map so that the number of actions for each partition
	//is q-1, so that the stay-in-place action is not included
	partitions = NumCuda<int>(NumActions, 2);
	int act = 0;
	for (int p = 0; p<nPartitions; p++){
		for (int Q = 0; Q<qSizes(p)-1; Q++){
			partitions(act, 0) = p;
			partitions(act, 1) = Q;
			// printf("#%i P: %i, Q: %i\n",act,p,Q);
			act++;
		}
	}

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

	//pre-build update tracker lists.  Mostly copying over from the sparse kmap:
	// for (int i = 0; i < nNHPPs; i++){
	// 	UpdateTracker(i, 0) = 0; //number of update actions stored here
	// 	NumCuda<int> connected_spins(qSizes.dims[0]);
	// 	for (int j = 0; j<=qSizes.dims[0]; j++) connected_spins(j) = 0;
	// 	for (int c = 1; c < kmap(i, 0, 0); c++) connected_spins(kmap(i,c,2)) = 1;
	// 	connected_spins(i)
	// 	for (int j = 0; j<=qSizes.dims[0]; j++){
	// 		if (connected_spins(j)){
	// 			for (int q = qCumulative(j)-qSizes(j); q<qCumulative(j); q++){
	// 				UpdateTracker(i,0)++;
	// 				UpdateTracker(UpdateTracker(i,0)) = q;
	// 			}
	// 		}
	// 	}
	// }


	// if (USE_GPU) UpdateTracker.CopyHostToDevice();
	if (USE_GPU) {
		sparse_kernels.CopyHostToDevice();
		partitions.CopyHostToDevice();
	}

}

__h__ __d__ float PottsPrecomputeBase::EnergyOfState(int* state){
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

__h__ __d__ void PottsPrecomputeBase::BeginEpoch(int iter){

	//intialize total energies to reflect the states that were just passed
	current_e = EnergyOfState(MiWrk);
	lowest_e = EnergyOfState(MiBest);

	recentUpdates = &UpdateTracker(Rep, 0);

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

__h__ __d__ void PottsPrecomputeBase::FinishEpoch(){
	float eps = 0.01;
	float e = EnergyOfState(MiWrk);
	if (current_e + eps < e || current_e - eps > e)
		printf("Working Energy tracking error: actual=%.5f, tracked=%.5f\n", e, current_e); 
}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__h__ __d__ float PottsPrecomputeBase::RealDE(int action_num){
	int action_partition = partitions(action_num, 0);
	int new_Mi = partitions(action_num, 1);
	int old_Mi = MiWrk[action_partition];
	if (new_Mi >= old_Mi) new_Mi++;
	int old_NHPP = qCumulative(action_partition)-qSizes(action_partition) + old_Mi;
	int new_NHPP = qCumulative(action_partition)-qSizes(action_partition) + new_Mi;
	// printf("Action: (M=%i, q=%i, pE=%.2f) -> (M=%i, q=%i, pE=%.2f)\n",
		// action_partition, old_Mi, NHPP_potentials(Rep, old_NHPP), action_partition, new_Mi, NHPP_potentials(Rep, action_num));
    return NHPP_potentials(Rep, new_NHPP) - NHPP_potentials(Rep, old_NHPP);
}

__h__ __d__ float PottsPrecomputeBase::PE(int action_num){
    int action_partition = partitions(action_num, 0);
	int new_Mi = partitions(action_num, 1);
	int old_Mi = MiWrk[action_partition];
	if (new_Mi >= old_Mi) new_Mi++;
	// int old_NHPP = qCumulative(action_partition)-qSizes(action_partition) + old_Mi;
	int new_NHPP = qCumulative(action_partition)-qSizes(action_partition) + new_Mi;
    return NHPP_potentials(Rep, new_NHPP);
}


//changes internal state to reflect the annealing step that was taken
__h__ __d__ void PottsPrecomputeBase::TakeAction_tic(int action_num){
	float dE = RealDE(action_num);
	// printf("taking action %i with dE %.5f\n", action_num, dE);
	current_e += dE;

	int j = partitions(action_num, 0);
	old_Mi = MiWrk[j]; //this needs to be determined here in tic, so that all threads have the correct old_Mi before MiWrk is set to the new Mi
}

__h__ __d__ void PottsPrecomputeBase::TakeAction_toc(int action_num){
	if (action_num < 0) return;

	int j = partitions(action_num, 0);
	int new_Mi = partitions(action_num, 1);
	if (new_Mi >= old_Mi) new_Mi++;
	int old_NHPP = qCumulative(j)-qSizes(j) + old_Mi;
	MiWrk[j] = new_Mi;
        
    // if (current_e < lowest_e){
    //     lowest_e = current_e;
    //     for (int i = Thrd; i < nPartitions; i+=nThrds) MiBest[i] = MiWrk[i];
    // }


	//initialize recent updates list with adding all states of the changed Potts spin:
	nRecentUpdates = 0; //have to be careful, becuase these point in the action space, not NHPP space
	for (int uu = 0; uu < qSizes(j)-1; uu++){
		recentUpdates[nRecentUpdates] = qCumulative(j)-qSizes(j)+uu-j;
		nRecentUpdates++;
	}
	

    for (int c = 1; c < kmap(j, 0, 0); c++){
		int i = kmap(j,c,2); //index of the connected Potts node
		if (i%nThrds != Thrd) continue; //each worker thread is only allowed to work on a particular subset of the partitions
		float w = kmap(j,c,1); //scalar multiplier
		int k = kmap(j,c,0); //which kernel
		int NHPP = qCumulative(i)-qSizes(i);

		//setup for tracking which actions have been updated.
		//Within each Potts nodes, updates are either sparse or dense,
		//depending on if the current action affects the energy of the active state of spin i
		bool dense_effect = false;
		int sparse_incriment_cnt = nRecentUpdates;
		if (qSizes(i)*2+nRecentUpdates > nNHPPs || nRecentUpdates < 0) {
			//becuase the kernel structure also allows mutiple kernels for the same connection,
			//we also need to prevent nRecentUpdates from ever exceeding NumActions.
			//it should also exactly equal NumActions, so that no actions are missed.
			nRecentUpdates = -1;
			sparse_incriment_cnt = 0;
		}
		
		for (int d = 1; d < sparse_kernels(k, old_Mi, 0, 0); d++){
			int q = sparse_kernels(k, old_Mi, d, 0);
			float ww = w*sparse_kernels(k, old_Mi, d, 1);
			NHPP_potentials(Rep, NHPP+q) -= ww;
			dense_effect = dense_effect | (q == MiWrk[i]);
			recentUpdates[sparse_incriment_cnt] = NHPP+q-i;
			if (q > MiWrk[i]) recentUpdates[sparse_incriment_cnt]--;
			sparse_incriment_cnt++;
		}
		for (int d = 1; d < sparse_kernels(k, new_Mi, 0, 0); d++){
			int q = sparse_kernels(k, new_Mi, d, 0);
			float ww = w*sparse_kernels(k, new_Mi, d, 1);
			NHPP_potentials(Rep, NHPP+q) += ww;
			dense_effect = dense_effect | (q == MiWrk[i]);
			recentUpdates[sparse_incriment_cnt] = NHPP+q-i;
			if (q > MiWrk[i]) recentUpdates[sparse_incriment_cnt]--;
			sparse_incriment_cnt++;
		}

		if (dense_effect && nRecentUpdates > 0){
			for (int uu = 0; uu < qSizes(i)-1; uu++){
				recentUpdates[nRecentUpdates] = qCumulative(i)-qSizes(i)+uu-i;
				nRecentUpdates++;
			}
		}
		else if (nRecentUpdates > 0){
			//otherwise, the updates were already accounted for in the above code,
			//and we just need to make it official:
			nRecentUpdates = sparse_incriment_cnt;
		}
    }
}


//derived classes with minor adjustments:
__h__ PottsPrecomputeAnnealable::PottsPrecomputeAnnealable(PyObject *task, int nReplicates, bool USE_GPU):
	PottsPrecomputeBase(task, nReplicates, USE_GPU) {
	if (USE_GPU) dispatch = GpuPottsPrecomputeDispatch;
	else         dispatch = CpuPottsPrecomputeDispatch;
}
	
__h__ __d__ float PottsPrecomputeAnnealable::GetActionDE(int action_num){
	return RealDE(action_num);
}

__h__ PottsPrecomputePEAnnealable::PottsPrecomputePEAnnealable(PyObject *task, int nReplicates, bool USE_GPU):
	PottsPrecomputeBase(task, nReplicates, USE_GPU) {
	if (USE_GPU) dispatch = GpuPottsPrecomputePEDispatch;
	else         dispatch = CpuPottsPrecomputePEDispatch;
}

__h__ __d__ float PottsPrecomputePEAnnealable::GetActionDE(int action_num){
	return PE(action_num);
}
