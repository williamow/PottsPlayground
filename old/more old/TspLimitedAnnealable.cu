#include <stdio.h>
#include "jsf.hpp"

class TspLimitedAnnealable
{
public:
	NumCuda<float> distances;
	NumCuda<int> ActionMap;

	jsf32na RngInt;
	
	int* MiBest;
	int* MiWrk;

	int nCities;

	float current_e;
	float lowest_e;
	int NumActions;
	int ActionOffset;

	int nThrds = 1; //number of threads working cooperatively
	int Thrd = 0; //index of this thread within the thread group
	int Rep;

	__host__ __device__ void SetIdentity(int thread_num, int total_threads, int replicate_){
		Thrd = thread_num;
		nThrds = total_threads;
		Rep = replicate_;
	}

	//=====================================================================================constructor methods
	TspLimitedAnnealable(NumCuda<float> &distances_, int nReplicates){
		distances = distances_;

		nCities = distances.dims[0];

		//allowable actions are swapping two cities.  nCities options for the first city, nCities options for the second.
		NumActions = nCities*1;
		ActionMap = NumCuda<int>(nReplicates, NumActions);

		for (int Rep = 0; Rep < nReplicates; Rep++)
			for (int i = 0; i<NumActions; i++) ActionMap(Rep, i) = i; //lame init

		ActionMap.CopyHostToDevice();

		ActionOffset = 0;

		// RngGenerator = std::minstd_rand();

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

	    RngInt = jsf32na(Thrd+1000*Rep); //seed with thread number

	}


	__host__ __device__ void FinalizeState(){
		
	}

	// ===================================================================================annealing methods
	//how much the total energy will change if this action is taken
	__host__ __device__ float GetActionDE(int action_num){
		action_num = ActionMap(Rep, action_num);
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
		action_num = ActionMap(Rep, action_num);
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
        current_e += GetActionDE(action_num);
	}

	__host__ __device__ void TakeAction_toc(int action_num){
		if (action_num >= 0){
			action_num = ActionMap(Rep, action_num);
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

		//stochastically change which actions are allowable at each iteration
    	for (int u = 0; u<4; u++) ActionMap(Rep, RngInt()%NumActions) = RngInt()%(nCities*nCities);
	}



};