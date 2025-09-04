#include <stdio.h>
#include <random>
#include "Annealables.h"


//=====================================================================================constructor methods
SwapAnnealable::SwapAnnealable(PyObject *task, bool USE_GPU){
	distances = NumCuda<float>(task, "distances", 2, false, USE_GPU);

	nCities = distances.dims[0];
	nPartitions = nCities;

	
	//in addition to swapping two cities, can also flip entire segments of a tour.
	// if (extended_actions) NumActions = nCities*nCities*2;
	// //swapping two cities:  nCities options for the first city, nCities options for the second.
	// else 
	NumActions = nCities*nCities;

	if (USE_GPU) dispatch = GpuSwapDispatch;
    else         dispatch = CpuSwapDispatch;

}

void SwapAnnealable::InitializeState(NumCuda<int> BestStates, NumCuda<int> WrkStates){
	
	int nReplicates = BestStates.dims[0];
	for (int replicate = 0; replicate < nReplicates; replicate++){
		for (int city = 0; city < nCities; city++){
	        BestStates(replicate, city) = city;
	        WrkStates(replicate, city) = city;
	    }
	}
}

__h__ __d__ void SwapAnnealable::BeginEpoch(int iter){

	//intialize total energies to reflect the states that were just passed
	current_e = 0;
    lowest_e = 0;
    for (int city = 0; city < nCities; city++){
    	current_e += distances(MiWrk[city], MiWrk[(city+1)%nCities]);
    	lowest_e +=  distances(MiBest[city], MiBest[(city+1)%nCities]);
    }
}


__h__ __d__ void SwapAnnealable::FinishEpoch(){
	
}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__h__ __d__ float SwapAnnealable::GetActionDE(int action_num){
	int action_type = action_num/(nCities*nCities);
	action_num = action_num%(nCities*nCities);
	int pos1 = action_num%nCities;
	int pos2 = action_num/nCities;

	//re-order, so pos2 is always greater than pos 1:
	if (pos2 == pos1) return 0;
	else if (pos2 < pos1){
		int temp = pos2;
		pos2 = pos1;
		pos1 = temp;
	}
	if (pos2 == nCities-1 && pos1==0) return 0; //just don't allow this.

	int city1 = MiWrk[pos1];
	int city2 = MiWrk[pos2];

	//for modulo protection
	pos1 += nCities;
	pos2 += nCities;

	int city1_left = MiWrk[(pos1-1)%nCities];
	int city1_right = MiWrk[(pos1+1)%nCities];
	int city2_left = MiWrk[(pos2-1)%nCities];
	int city2_right = MiWrk[(pos2+1)%nCities];

	//in this case action type 0 and 1 are the same,
	//and the normal type 0 dE calculation is erroneous, so we use the type 1 dE calculation
	if (pos2-pos1 == 1 || pos2-pos1 == nCities-1) action_type = 1; 
	float dE = 0;

	// if action_type is 0, then just swap cities; if 1, then reverse a segment
	if (action_type == 0){
		dE += distances(city2, city1_left);
		dE += distances(city2, city1_right);

		dE += distances(city1, city2_left);
		dE += distances(city1, city2_right);

		dE -= distances(city2, city2_left);
		dE -= distances(city2, city2_right);

		dE -= distances(city1, city1_left);
		dE -= distances(city1, city1_right);

	} else if (action_type == 1){
		//if a segment is reversed, two distances are removed and two new ones are added.
		//the distances between cities internal to the segment do not change.
		dE -= distances(city1, city1_left);
		dE -= distances(city2, city2_right);

		dE += distances(city1, city2_right);
		dE += distances(city2, city1_left);
	}
	
    return dE;
} 

//changes internal state to reflect the annealing step that was taken
__h__ __d__ void SwapAnnealable::TakeAction_tic(int action_num){
    current_e += GetActionDE(action_num);
    // if (current_e < lowest_e) lowest_e = current_e;
}

__h__ __d__ void SwapAnnealable::TakeAction_toc(int action_num){
	if (action_num < 0) return;
	if (Thrd != 0) return; //no parallel processing enabled in this step

	int action_type = action_num/(nCities*nCities);
	action_num = action_num%(nCities*nCities);
	int pos1 = action_num%nCities;
	int pos2 = action_num/nCities;

	//re-order, so pos2 is always greater than pos 1:
	if (pos2 == pos1) return; //no action
	else if (pos2 < pos1){
		int temp = pos2;
		pos2 = pos1;
		pos1 = temp;
	}
	if (pos2 == nCities-1 && pos1==0) return; //just don't allow this.

	if (action_type == 0){
		//swap the two cites
		int city1 = MiWrk[pos1];
		int city2 = MiWrk[pos2]; 

		MiWrk[pos2] = city1;
		MiWrk[pos1] = city2;
	}
	else if (action_type == 1){
		//reverse the segment of cities between and including pos1 and pos2
		while (pos1 < pos2){
			int city1 = MiWrk[pos1];
			int city2 = MiWrk[pos2]; 

			MiWrk[pos2] = city1;
			MiWrk[pos1] = city2;

			pos1++;
			pos2--;
		}

	}
    // printf("Current: %.2f, Lowest: %.2f\n", current_e, lowest_e);
    // if (current_e < lowest_e){
    //     lowest_e = current_e;
    //     for (int i = 0; i < nCities; i++) MiBest[i] = MiWrk[i];
    // }

	// current_e = 0;
    // lowest_e = 0;
    // for (int city = 0; city < nCities; city++){
    	// current_e += distances(MiWrk[city], MiWrk[(city+1)%nCities]);
    	// lowest_e +=  distances(MiBest[city], MiBest[(city+1)%nCities]);
    // }
}