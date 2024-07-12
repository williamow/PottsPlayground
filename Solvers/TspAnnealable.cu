#include <stdio.h>
#include <random>
#include "Annealables.h"

//=====================================================================================constructor methods
TspAnnealable::TspAnnealable(PyObject *task, bool USE_GPU){
	distances = NumCuda<float>(task, "distances", 2, false, USE_GPU);

	nCities = distances.dims[0];

	//allowable actions are swapping two cities.  nCities options for the first city, nCities options for the second.
	NumActions = nCities*nCities;

	if (USE_GPU) dispatch = GpuDispatch<TspAnnealable>;
    else         dispatch = CpuDispatch<TspAnnealable>;

}

__host__ __device__ void TspAnnealable::BeginEpoch(int iter){

	if (iter==0){
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


__host__ __device__ void TspAnnealable::FinishEpoch(){
	
}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__host__ __device__ float TspAnnealable::GetActionDE(int action_num){
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

//changes internal state to reflect the annealing step that was taken
__host__ __device__ void TspAnnealable::TakeAction_tic(int action_num){
    current_e += GetActionDE(action_num);
}

__host__ __device__ void TspAnnealable::TakeAction_toc(int action_num){
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