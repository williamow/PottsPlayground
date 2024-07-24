#include <stdio.h>
#include <random>

class TspAnnealable
{
public:
	NumCuda<float> distances;
	
	int* MiBest;
	int* MiWrk;

	int nCities;

	float current_e;
	float lowest_e;
	int NumActions;

	int nThrds = 1; //number of threads working cooperatively
	int Thrd = 0; //index of this thread within the thread group
	int Rep;

	__host__ __device__ void SetIdentity(int thread_num, int total_threads, int replicate_){
		Thrd = thread_num;
		nThrds = total_threads;
		Rep = replicate_;
	}

	//=====================================================================================constructor methods
	TspAnnealable(NumCuda<float> &distances_){
		distances = distances_;

		nCities = distances.dims[0];

		//allowable actions are swapping two cities.  nCities options for the first city, nCities options for the second.
		NumActions = nCities*nCities;

	}

	__host__ __device__ void SetState(int* &WrkState_, int* &BestState_, bool initialize){
		MiBest = BestState_;
		MiWrk = WrkState_;

		if (initialize){
			for (int city = 0; city < nCities; city++){
	            MiWrk[city] = city;
	            MiBest[city] = MiWrk[city];
	        }
		}

		//intialize total energies to reflect the states that were just passed
		current_e = 0;
	    lowest_e = 0;
	    for (int city = 0; city < nCities; city++){
	    	current_e += distances(MiWrk[city], MiWrk[(city+1)%nCities]);
	    	lowest_e +=  distances(MiBest[city], MiBest[(city+1)%nCities]);
	    }
	}


	__host__ __device__ void FinalizeState(){
		
	}

	// ===================================================================================annealing methods
	//how much the total energy will change if this action is taken
	__host__ __device__ float GetActionDE(int action_num){
		int pos1 = action_num%nCities;
		int pos2 = action_num/nCities;

		int city1 = MiWrk[pos1];
		int city2 = MiWrk[pos2];

		//for modulo protection
		pos1 += nCities;
		pos2 += nCities;

		float dE = 0;

		//different alg depending if city swaps are adjacent or not:
		if ((pos1-pos2+nCities)%nCities == 1){
			// dE += distances(city2, MiWrk[(pos1-1)%nCities]);
			dE += distances(city2, MiWrk[(pos1+1)%nCities]);

			dE += distances(city1, MiWrk[(pos2-1)%nCities]);
			// dE += distances(city1, MiWrk[(pos2+1)%nCities]);

			dE -= distances(city2, MiWrk[(pos2-1)%nCities]);
			// dE -= distances(city2, MiWrk[(pos2+1)%nCities]);

			// dE -= distances(city1, MiWrk[(pos1-1)%nCities]);
			dE -= distances(city1, MiWrk[(pos1+1)%nCities]);
		}
		else if ((pos2-pos1+nCities)%nCities == 1){
			dE += distances(city2, MiWrk[(pos1-1)%nCities]);
			// dE += distances(city2, MiWrk[(pos1+1)%nCities]);

			// dE += distances(city1, MiWrk[(pos2-1)%nCities]);
			dE += distances(city1, MiWrk[(pos2+1)%nCities]);

			// dE -= distances(city2, MiWrk[(pos2-1)%nCities]);
			dE -= distances(city2, MiWrk[(pos2+1)%nCities]);

			dE -= distances(city1, MiWrk[(pos1-1)%nCities]);
			// dE -= distances(city1, MiWrk[(pos1+1)%nCities]);
		}
		else{
			dE += distances(city2, MiWrk[(pos1-1)%nCities]);
			dE += distances(city2, MiWrk[(pos1+1)%nCities]);

			dE += distances(city1, MiWrk[(pos2-1)%nCities]);
			dE += distances(city1, MiWrk[(pos2+1)%nCities]);

			dE -= distances(city2, MiWrk[(pos2-1)%nCities]);
			dE -= distances(city2, MiWrk[(pos2+1)%nCities]);

			dE -= distances(city1, MiWrk[(pos1-1)%nCities]);
			dE -= distances(city1, MiWrk[(pos1+1)%nCities]);
		}
		
        return dE;
	} 

	//the potential energy; only really makes sense for Potts models.  Important only for my particular research purposes.
	__host__ __device__ float GetActionPE(int action_num){
		int pos1 = action_num%nCities;
		int pos2 = action_num/nCities;

		int city1 = MiWrk[pos1];
		int city2 = MiWrk[pos2];

		//for modulo protection
		pos1 += nCities;
		pos2 += nCities;

		float pE = 0;
		pE += distances(city2, MiWrk[pos1-1]);
		pE += distances(city2, MiWrk[pos1+1]);

		pE += distances(city1, MiWrk[pos2-1]);
		pE += distances(city1, MiWrk[pos2+1]);
		
        return pE;
	} 

	//changes internal state to reflect the annealing step that was taken
	__host__ __device__ void TakeAction_tic(int action_num){
		// int action_partition = partitions(action_num);
		// int action_Mi = partition_states(action_num);

        // if (action_num == last_action) current_e += last_dE;
        // else 
        current_e += GetActionDE(action_num); //super inneficient, fix! pass dE into here somehow...

        // MiWrk[action_partition] = action_Mi; //needs to happen after all the GetActionDE calls
            
        // if (current_e < lowest_e){
        //     lowest_e = current_e;
        //     for (int i = 0; i < nPartitions; i++) MiBest[i] = MiWrk[i];
        // }
	}

	__host__ __device__ void TakeAction_toc(int action_num){
		if (action_num < 0) return;
		
		int pos1 = action_num%nCities;
		int pos2 = action_num/nCities;

		int city1 = MiWrk[pos1];
		int city2 = MiWrk[pos2]; 

		MiWrk[pos2] = city1;
		MiWrk[pos1] = city2;
            
        if (current_e < lowest_e){
            lowest_e = current_e;
            for (int i = 0; i < nCities; i++) MiBest[i] = MiWrk[i];
        }
	}



};