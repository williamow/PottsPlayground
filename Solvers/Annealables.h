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
#include "NumCuda.hpp"
#include "PieceWiseLinear.cpp"
#include "AnnealingCore.cu"

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

	__host__ __device__ void SetIdentity(int thread_num, int total_threads, int replicate_, int* WrkState_, int* BestState_){
		Thrd = thread_num;
		nThrds = total_threads;
		Rep = replicate_;
		MiBest = BestState_;
		MiWrk = WrkState_;
	}

	DispatchFunc* dispatch; //defined in AnnealingCore.cu
};


class PottsJitAnnealable: public Annealable {
private:
	NumCuda<float> kmap;
	NumCuda<int> qSizes;
	NumCuda<int> partitions;
	NumCuda<int> partition_states;
	NumCuda<float> kernels;
	NumCuda<float> biases;
public:
	__host__ PottsJitAnnealable(PyObject* task, bool USE_GPU);
	__host__ __device__ void BeginEpoch(int iter);
	__host__ __device__ void FinishEpoch();
	__host__ __device__ float GetActionDE(int action_num);
	__host__ __device__ void TakeAction_tic(int action_num);
	__host__ __device__ void TakeAction_toc(int action_num);
};

class PottsPrecomputeAnnealable: public Annealable{
private:
	NumCuda<int> kmap;
	NumCuda<int> qSizes;
	NumCuda<int> qCumulative;
	NumCuda<int> partitions;
	NumCuda<int> partition_states;
	NumCuda<float> kernels;
	NumCuda<float> NHPP_potentials;
	//for storing info between tic and toc action phases:
	int action_partition;
	int old_Mi;
	int new_Mi; 
public:
	__host__ PottsPrecomputeAnnealable(PyObject *task, int nReplicates, bool USE_GPU);
	__host__ __device__ void BeginEpoch(int iter);
	__host__ __device__ void FinishEpoch();
	__host__ __device__ float GetActionDE(int action_num);
	__host__ __device__ void TakeAction_tic(int action_num);
	__host__ __device__ void TakeAction_toc(int action_num);
};

class TspAnnealable: public Annealable {
private:
	int nCities;
	NumCuda<float> distances;
public:
	__host__ TspAnnealable(PyObject *task, bool USE_GPU);
	__host__ __device__ void BeginEpoch(int iter);
	__host__ __device__ void FinishEpoch();
	__host__ __device__ float GetActionDE(int action_num);
	__host__ __device__ void TakeAction_tic(int action_num);
	__host__ __device__ void TakeAction_toc(int action_num);
};

#endif //ANNEALLABLES_INCLUDED