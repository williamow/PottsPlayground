#ifndef ANNEALABLES_INCLUDED
#define ANNEALABLES_INCLUDED

//python includes:
#define PY_SSIZE_T_CLEAN //why? not sure.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION //makes old API unavailable, and supresses API depreciation warning
#include <Python.h>
#include <numpy/ndarrayobject.h>
//regular includes:
#include <stdio.h>
#include <random>
//my includes:
#include "NumCuda.h"
#include "PieceWiseLinear.h"
#include "Cores.h"

//__host__ and __device__ tell nvcc which functions are to be compiled for cpu/gpu respectively.
//however C++ compilers know nothing of this, so this does some of our own preprocessing so cpu-only compilation is possible.
#ifdef __CUDACC__
    #define __h__ __host__
    #define __d__ __device__ 
#else
    #define __h__
    #define __d__
#endif

//defines format of the dispatch functions, for creating function pointers more easily
using DispatchFunc = void(
		void*, //the task, which must be specified as a void pointer since the actual type changes but I want an unchanging function interface
		DispatchArgs &);

		// NumCuda<int> &, NumCuda<int> &, //working states, best states
		// NumCuda<float> &, NumCuda<float> &, //working energies, best energies
		// NumCuda<float> &, //communal space memory for interthread cooperation
		// NumCuda<int> &, //For returning the total number of actual flips
		// PieceWiseLinear, int, int, int, //annealing temperature specification, nOptions, nActions, nWorkers
		// long, long, //min iter, max iter
		// float, //termination threshold energy
		// NumCuda<int> &//global halt
		// );

class Annealable{ //full definition of parent class; not a fully functional object, just helps to define some of the data fields
public:
	int* MiBest;
	int* MiWrk;

	int nNHPPs;
	int nPartitions;

	float current_e;
	float lowest_e;
	int NumActions;


	int nThrds = 1; //number of threads working cooperatively
	int Thrd = 0; //index of this thread within the thread group
	int Rep;

	//for pre-compute behavior.  Not functional, but allows all annealables to compile:
	int nRecentUpdates = -1;
	int* recentUpdates;

	__h__ __d__ void SetIdentity(int thread_num, int total_threads, int replicate_, int* WrkState_, int* BestState_){
		Thrd = thread_num;
		nThrds = total_threads;
		Rep = replicate_;
		MiBest = BestState_;
		MiWrk = WrkState_;
	}

	__h__ __d__ void check_energy(){
		if (current_e < lowest_e){
        	lowest_e = current_e;
        	for (int i = 0; i < nPartitions; i++) MiBest[i] = MiWrk[i];
    	}
	}

	__h__ virtual void InitializeState(NumCuda<int> BestStates, NumCuda<int> WrkStates){
		//dumb default initialization works for most Annealables, but not all.
		//For instance TspAnnealable has special requirements, and has an overriding function.
		int nPartitions = BestStates.dims[1];
		int nReplicates = BestStates.dims[0];
		for (int replicate = 0; replicate < nReplicates; replicate++){
			for (int partition = 0; partition < nPartitions; partition++){
	            BestStates(replicate, partition) = 0;
	            WrkStates(replicate, partition) = 0;
	        }
	    }
	}

	DispatchFunc* dispatch;
};

extern DispatchFunc* GpuIsingDispatch;
extern DispatchFunc* CpuIsingDispatch;
class IsingAnnealable: public Annealable {
private:
	NumCuda<float> kmap;
	NumCuda<float> biases;
	int Mi_proposed_holder;
public:
	__h__ IsingAnnealable(PyObject* task, bool USE_GPU);
	__h__ __d__ float EnergyOfState(int* state);
	__h__ __d__ void BeginEpoch(int iter);
	__h__ __d__ void FinishEpoch();
	__h__ __d__ float GetActionDE(int action_num);
	__h__ __d__ void TakeAction_tic(int action_num);
	__h__ __d__ void TakeAction_toc(int action_num);
};

extern DispatchFunc* GpuIsingPrecomputeDispatch;
extern DispatchFunc* CpuIsingPrecomputeDispatch;
class IsingPrecomputeAnnealable: public Annealable {
private:
	NumCuda<float> kmap;
	NumCuda<float> biases;
	int Mi_proposed_holder;
	NumCuda<double> NHPP_potentials;
	NumCuda<int> UpdateIndices;
public:
	__h__ IsingPrecomputeAnnealable(PyObject* task, int nReplicates, bool USE_GPU);
	__h__ __d__ float EnergyOfState(int* state);
	__h__ __d__ void BeginEpoch(int iter);
	__h__ __d__ void FinishEpoch();
	__h__ __d__ float GetActionDE(int action_num);
	__h__ __d__ void TakeAction_tic(int action_num);
	__h__ __d__ void TakeAction_toc(int action_num);
};


extern DispatchFunc* GpuPottsJitDispatch;
extern DispatchFunc* CpuPottsJitDispatch;
class PottsJitAnnealable: public Annealable {
private:
	NumCuda<float> kmap;
	NumCuda<int> qSizes;
	NumCuda<int> partitions;
	NumCuda<int> partition_states;
	NumCuda<float> kernels;
	NumCuda<float> biases;
public:
	__h__ PottsJitAnnealable(PyObject* task, bool USE_GPU);
	__h__ __d__ float EnergyOfState(int* state);
	__h__ __d__ void BeginEpoch(int iter);
	__h__ __d__ void FinishEpoch();
	__h__ __d__ float GetActionDE(int action_num);
	__h__ __d__ void TakeAction_tic(int action_num);
	__h__ __d__ void TakeAction_toc(int action_num);
};

//class that pre-computes all possible delta-E energy changes after each update, and keeps track of all the possibilties
//in a separate memory location.
//Designed to be faster in some cases when full parallel-hypothesis sampling is used.
class PottsPrecomputeBase: public Annealable{
private:
	NumCuda<float> kmap;
	NumCuda<int> qSizes;
	NumCuda<int> qCumulative;
	NumCuda<int> partitions;
	NumCuda<int> partition_states;
	NumCuda<float> dense_kernels;
	NumCuda<float> sparse_kernels;
	NumCuda<double> NHPP_potentials;
	NumCuda<int> UpdateTracker;
	NumCuda<float> biases;
	int old_Mi; //for passing data from tic phase to toc phase
public:
	__h__ PottsPrecomputeBase(PyObject *task, int nReplicates, bool USE_GPU);
	__h__ __d__ float EnergyOfState(int* state);
	__h__ __d__ void BeginEpoch(int iter);
	__h__ __d__ void FinishEpoch();
	__h__ __d__ float RealDE(int action_num);
	__h__ __d__ float PE(int action_num);
	__h__ __d__ void TakeAction_tic(int action_num);
	__h__ __d__ void TakeAction_toc(int action_num);
};

//Potts pre-compute is split into two variants, which are almost exactly the same.
//the difference is that on uses potential energy of each action, rather than the energy change,
//which is much less efficient (so you shouldn't use it) but unfortunately I originally designed hardware
//that used the potential energy method, and published results about it, so I wanted to have the comparison
//between that bad method and proper DE-based caluclations in this suite of simulation tools.
extern DispatchFunc* GpuPottsPrecomputeDispatch;
extern DispatchFunc* CpuPottsPrecomputeDispatch;
class PottsPrecomputeAnnealable: public PottsPrecomputeBase{
public:
	__h__ PottsPrecomputeAnnealable(PyObject *task, int nReplicates, bool USE_GPU);
	__h__ __d__ float GetActionDE(int action_num);
};

extern DispatchFunc* GpuPottsPrecomputePEDispatch;
extern DispatchFunc* CpuPottsPrecomputePEDispatch;
class PottsPrecomputePEAnnealable: public PottsPrecomputeBase{
public:
	__h__ PottsPrecomputePEAnnealable(PyObject *task, int nReplicates, bool USE_GPU);
	__h__ __d__ float GetActionDE(int action_num);
};

extern DispatchFunc* GpuTspDispatch;
extern DispatchFunc* CpuTspDispatch;
class TspAnnealable: public Annealable {
private:
	int nCities;
	NumCuda<float> distances;
public:
	__h__ TspAnnealable(PyObject *task, bool USE_GPU, bool extended_actions);
	__h__ void InitializeState(NumCuda<int> BestStates, NumCuda<int> WrkStates) override;
	__h__ __d__ void BeginEpoch(int iter);
	__h__ __d__ void FinishEpoch();
	__h__ __d__ float GetActionDE(int action_num);
	__h__ __d__ void TakeAction_tic(int action_num);
	__h__ __d__ void TakeAction_toc(int action_num);
};

#endif //ANNEALLABLES_INCLUDED