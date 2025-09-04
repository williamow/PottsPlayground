#include <stdio.h>
#include <random>
#include "Annealables.h"

//A potts model, but with spin exchange updates.  There are two variants:
//one with only spin exchanges,
//and one with both spin exchanges and regular, single spin flips.
//When only spin exchanges are allowed, identical behavior to old TspAnnealable model.

//=====================================================================================constructor methods
SwapAnnealable::SwapAnnealable(PyObject *task, bool USE_GPU, bool extended_actions) : PottsJitAnnealable(task, USE_GPU) {
	
	qCumulative = 		NumCuda<int>(task, "qCumulative", 1, false, USE_GPU);

	if (extended_actions)
		NumActions = nNHPPs + nPartitions*nPartitions; //swapping and regular Potts updates
	else
		NumActions = nPartitions*nPartitions; //swapping only

	if (USE_GPU) dispatch = GpuSwapDispatch;
    else         dispatch = CpuSwapDispatch;

}

void SwapAnnealable::InitializeState(NumCuda<int> BestStates, NumCuda<int> WrkStates){
	
	int nReplicates = BestStates.dims[0];
	for (int replicate = 0; replicate < nReplicates; replicate++){
		for (int spin = 0; spin < nPartitions; spin++){
	        BestStates(replicate, spin) = spin%qSizes(spin);
	        WrkStates(replicate, spin) = spin%qSizes(spin);
	    }
	}
}

// ===================================================================================annealing methods
//how much the total energy will change if this action is taken
__h__ __d__ float SwapAnnealable::GetActionDE(int action_num){
	if (action_num >= nPartitions*nPartitions){
		//extended action, i.e. regular PottsJit edit
		action_num = action_num - nPartitions*nPartitions;
		return PottsJitAnnealable::GetActionDE(action_num);
	}
	else{
		//swap action:
		int spin1 = action_num%nPartitions;
		int spin2 = action_num/nPartitions;

		//essentially, two simultaneous Potts spin updates:
		int spin1_newq = MiWrk[spin2]%qSizes(spin1); //modulo for overflow protection
		int spin2_newq = MiWrk[spin1]%qSizes(spin2); //modulo for overflow protection

		int fake_action_1 = qCumulative(spin1)-qSizes(spin1)+spin1_newq;
		float dE = PottsJitAnnealable::GetActionDE(fake_action_1);
		
		//now, must add to DE to account for the second spin flip:
		dE += biases(spin2, spin2_newq) - biases(spin2, MiWrk[spin2]);

		for (int c = 1; c < kmap(spin2,0,0); c++){
			int j = kmap(spin2,c,2); //index of the connected Potts node
			float w = kmap(spin2,c,1); //scalar multiplier
			int k = kmap(spin2,c,0); //which kernel
			int mj = MiWrk[j];
			if (j == spin1) //must take into account spin1's flip
				mj = spin1_newq;
			dE += w*kernels(k, spin2_newq, mj);
			dE -= w*kernels(k, MiWrk[spin2], mj);
		}

	    return dE;
	}

} 

//changes internal state to reflect the annealing step that was taken
__h__ __d__ void SwapAnnealable::TakeAction_tic(int action_num){
    current_e += GetActionDE(action_num);

    if (action_num < nPartitions*nPartitions){
    	int spin1 = action_num%nPartitions;
		int spin2 = action_num/nPartitions;

		//hold these values for multi-threading safety
		spin1_newq = MiWrk[spin2]%qSizes(spin1); //modulo for overflow protection
		spin2_newq = MiWrk[spin1]%qSizes(spin2); //modulo for overflow protection
    }
    
}

__h__ __d__ void SwapAnnealable::TakeAction_toc(int action_num){
	if (action_num < 0) return;
	
	if (action_num >= nPartitions*nPartitions){
		//extended action, i.e. regular PottsJit edit
		action_num = action_num - nPartitions*nPartitions;
		PottsJitAnnealable::TakeAction_toc(action_num);
	}
	else{
		//swap action:
		int spin1 = action_num%nPartitions;
		int spin2 = action_num/nPartitions;

		MiWrk[spin1] = spin1_newq;
		MiWrk[spin2] = spin2_newq;
	}
}