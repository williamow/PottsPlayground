#include <stdio.h>

//definitions to set up dual-use headers for CPU compilation only:
#define __h__
#define __d__

#include "NumCuda.h"
#include "Annealables.h" 

//these need to be included, as this file functions as the compiler entry point for them
#include "TspAnnealable.h"
#include "IsingAnnealable.h"
#include "IsingPrecomputeAnnealable.h"
#include "PottsJitAnnealable.h"
#include "PottsPrecomputeAnnealable.h"

#include "Cores.h"
bool GetGpuAvailability(){
	printf("This copy of PottsPlayground is not compiled with GPU support.\n");
    return false;
}

template class NumCuda<int>;
template class NumCuda<float>;

DispatchFunc* GpuTspDispatch = NULL;
DispatchFunc* GpuIsingDispatch = NULL;
DispatchFunc* GpuIsingPrecomputeDispatch = NULL;
DispatchFunc* GpuPottsJitDispatch = NULL;
DispatchFunc* GpuPottsPrecomputeDispatch = NULL;
DispatchFunc* GpuPottsPrecomputePEDispatch = NULL;
