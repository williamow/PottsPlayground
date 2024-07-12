#include <stdio.h>
#include <random>
#include <cuda_runtime.h>
#include <typeinfo>



	//=====================================================================================constructor methods
PottsPrecomputeAnnealable::PottsPrecomputeAnnealable(PyObject *task, int nReplicates, bool USE_GPU){}

	qSizes = 			NumCuda<int>(task, "qSizes", 1, false, USE_GPU);
	partitions = 		NumCuda<int>(task, "Partitions", 1, false, USE_GPU);
	partition_states = 	NumCuda<int>(task, "Partition_states", 1, false, USE_GPU);
	kmap = 				NumCuda<float>(task, "kmap_sparse", 3, false, USE_GPU);
	kernels = 			NumCuda<float>(task, "kernels", 3, false, USE_GPU);
	biases = 			NumCuda<float>(task, "biases", 2, false, USE_GPU);

		NumCuda<int> &kmap_,
						  NumCuda<int> &qSizes_, 
						  NumCuda<int> &qCumulative_,
						  NumCuda<int> &partitions_,
						  NumCuda<int> &partition_states_,
						  NumCuda<float> &kernels_,
						  int nReplicates){
		kmap = kmap_;
		qSizes = qSizes_;
		qCumulative = qCumulative_;
		partitions = partitions_;
		partition_states = partition_states_;
		kernels = kernels_;
		// printf("qSizes hd: %i\n", &qSizes(0));
		// printf("qSizes dims[0] %i\n", qSizes.dims[0]);

		nPartitions = qSizes.dims[0];
		nNHPPs = qSizes.sum();
		NumActions = nNHPPs;
		NHPP_potentials = NumCuda<float>(nReplicates, nNHPPs);

	}

	__host__ __device__ void PottsPrecomputeAnnealable::BeginEpoch(int iter){

		if (iter==0){
			for (int partition = 0; partition < nPartitions; partition++){
				// printf("qSizes[%i]=%i\n", partition, qSizes(partition));
	            MiWrk[partition] = partition%qSizes(partition); //"pseudorandom" initialization that is consistent between threads working together
	            MiBest[partition] = MiWrk[partition];
	        }
		}

		//intialize total energies to reflect the states that were just passed
		current_e = 0;
	    lowest_e = 0;
	    for (int i = 0; i < nPartitions; i++){
	        for (int j = 0; j < nPartitions; j++){
	            current_e += kernels(kmap(i,j), MiWrk[i], MiWrk[j]);//[   qMax * qMax *kmap(i, j)  +  MiWrk[i]*qMax  +  MiWrk[j]  ];
	            lowest_e += kernels(kmap(i,j), MiBest[i], MiBest[j]);
	            //weights contribute to the active energy when both sides of the weights are selected.
	        }
	    }
	    current_e = current_e / 2;
	    lowest_e = lowest_e / 2;

        // for (int NHPP=Thrd; NHPP<nNHPPs; NHPP+=nThrds) NHPP_potentials(Rep, NHPP) = 0;//init all to zero

        //need to make sure that the potentials this thread sets to zero are the same that it calculates the starting energy for
        for (int i=Thrd; i<nPartitions; i+=nThrds){
			int NHPP = qCumulative(i)-qSizes(i);
			for (int j=0; j<qSizes(i); j++){
				NHPP_potentials(Rep, NHPP) = 0;
				NHPP++;
			}
		}

        for (int h=0; h<nPartitions; h++){
        	for (int i=Thrd; i<nPartitions; i+=nThrds){
				int NHPP = qCumulative(i)-qSizes(i);
				for (int j=0; j<qSizes(i); j++){
					NHPP_potentials(Rep, NHPP) += kernels(kmap(h, i), MiWrk[h], j);
					NHPP++;
				}
			}
		}

		//check that NHPP potentials match the alternately calculated energy:

	}

	__host__ __device__ void PottsPrecomputeAnnealable::FinishEpoch(){
	}

	// ===================================================================================annealing methods
	//how much the total energy will change if this action is taken
	__host__ __device__ float PottsPrecomputeAnnealable::GetActionDE(int action_num){
		int action_partition = partitions(action_num);
		int old_Mi = MiWrk[action_partition];
		int old_NHPP = qCumulative(action_partition)-qSizes(action_partition) + old_Mi;
		// printf("old/new action # in get dE: %i, %i\n", old_NHPP, action_num);
		// printf("old/new potential in get dE: %.2f, %.2f\n",NHPP_potentials(Rep, old_NHPP), NHPP_potentials(Rep, action_num));
        return NHPP_potentials(Rep, action_num) - NHPP_potentials(Rep, old_NHPP);
	} 

	//changes internal state to reflect the annealing step that was taken
	__host__ __device__ void PottsPrecomputeAnnealable::TakeAction_tic(int action_num){
		// return;
		action_partition = partitions(action_num);
		new_Mi = partition_states(action_num);
		old_Mi = MiWrk[action_partition];
		int old_NHPP = qCumulative(action_partition)-qSizes(action_partition) + old_Mi;

		// printf("Old NHPP is: %i\n", old_NHPP);
		// printf("dE is: %.3f\n", NHPP_potentials(Rep, action_num) - NHPP_potentials(Rep, old_NHPP));

		current_e += NHPP_potentials(Rep, action_num) - NHPP_potentials(Rep, old_NHPP);

		

	}

	__host__ __device__ void PottsPrecomputeAnnealable::TakeAction_toc(int action_num){
		if (action_num < 0) return;
		MiWrk[action_partition] = new_Mi;
            
        if (current_e < lowest_e){
            lowest_e = current_e;
            for (int i = Thrd; i < nPartitions; i+=nThrds) MiBest[i] = MiWrk[i];
        }

    	//adjust all of the potential energies to reflect the new working state:
		for (int i=Thrd; i<nPartitions; i+=nThrds){
			int NHPP = qCumulative(i)-qSizes(i);
			for (int j=0; j<qSizes(i); j++){
				NHPP_potentials(Rep, NHPP) -= kernels(kmap(action_partition, i), old_Mi, j);
				NHPP_potentials(Rep, NHPP) += kernels(kmap(action_partition, i), new_Mi, j);
				NHPP++;
			}
		}
        
	}