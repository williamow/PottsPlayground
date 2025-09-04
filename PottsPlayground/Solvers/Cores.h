#ifndef CORES_INCLUDED
#define CORES_INCLUDED

#include "NumCuda.h"
#include "PieceWiseLinear.h"

//for use in the main annealing function:
//defined here, but has two alternate definitions in GpuCore and GpuCoreAlt, for Gpu and non-Gpu builds respectively.
bool GetGpuAvailability();

struct DispatchArgs {
	NumCuda<int> WrkStates;
	NumCuda<int> BestStates;
	NumCuda<float> WrkEnergies;
	NumCuda<float> BestEnergies;
	NumCuda<float> ActArb;
	NumCuda<int> RealFlips;
	PieceWiseLinear PwlTemp;
	int nOptions;
	int nActions;
	int nWorkers;
	long MinIter;
	long MaxIter;
	float e_th;
	NumCuda<int> GlobalHalt;
	const char *algo;
	bool ParallelUpdates;
	const char *pMode;
};

//little class for keeping track of which actions should be taken and which should be discarded.
class ActionArbitrator{

public:
NumCuda<float> *ActArb;
int nActions;
int worstActionOfTheBest;
float worstPriorityOfTheBest;
int replicate_index;

__h__ __d__ inline ActionArbitrator(NumCuda<float> * ActArb_, int replicate_index_){
	replicate_index = replicate_index_;
	ActArb = ActArb_;
	nActions = ActArb->dims[1];

	for (int act = 0; act<nActions; act++){
		//initialize each to the default Null action, with dE=0.
		(*ActArb)(replicate_index, act, 1) = -1; 
		// printf("Arb slot %i: Priority = %.3f, action = %i\n", act, (*ActArb)(replicate_index, act, 0), int((*ActArb)(replicate_index, act, 1)));
	}

	UpdateWorst();
}

__h__ __d__ inline void UpdateWorst(){
	//find which option is now the worst, to be booted first the next time there is an update
	worstActionOfTheBest = 0;
	worstPriorityOfTheBest = (*ActArb)(replicate_index, 0, 0);

	for (int act = 0; act<nActions; act++){
		if ((*ActArb)(replicate_index, act, 0) > worstPriorityOfTheBest){
			worstPriorityOfTheBest = (*ActArb)(replicate_index, act, 0);
			worstActionOfTheBest = act;
		}
	}
}

__h__ __d__ inline void AddOption(int NewOption, float NewPriority){
	// printf("New Priority = %.3f, New action = %i\n", NewPriority, NewOption);
	if ((*ActArb)(replicate_index, worstActionOfTheBest, 0) != worstPriorityOfTheBest) UpdateWorst(); //if another GPU thread made a change
	if (NewPriority < worstPriorityOfTheBest){
		//replace old worst option with the new option
		(*ActArb)(replicate_index, worstActionOfTheBest, 0) = NewPriority;
		(*ActArb)(replicate_index, worstActionOfTheBest, 1) = NewOption;
		// printf("Changing Arb slot %i to Priority = %.3f, action = %i\n", worstActionOfTheBest, NewPriority, NewOption);

		UpdateWorst();
	}
}
};

#endif //CORES_INCLUDED