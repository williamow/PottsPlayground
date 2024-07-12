#include <stdio.h>

class Annealable
{
public:
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
	//sets a pointer to a pre-allocated vector of ints that stores the states between annealing runs.
	//this happens inside each thread, since different threads will be working on different states.
	__host__ __device__ void SetState(int* &WrkState_, int* &BestState_, bool initialize){
		printf("Error! SetState not implimented in child class\n");
	}

	//in case a class wants to internally store the state in a non-standard format during annealing,
	//it can convert the state back at the end using this function
	__host__ __device__ void FinalizeState(){
	}

	//there is no shared constructor, since the construction requirements (i.e. number of arguements) will depend on the specifics of each subclass.

	//An annealable object may provide an interface that allows multiple threads to cooperatively anneal a single system.
	//in this case, the thread must tell the annealable object the details, so that different threads can update different parts of the state.
	//I expect most or all subclasses will not get around to taking advantage of this complexity.


	// __host__ __device__ virtual Annealable* clone(){
		// printf("Error! clone has not been implimented in child class\n");
		// return new Annealable(*this);
	// }

	// ===================================================================================annealing methods

	// //how much the total energy will change if this action is taken
	// __host__ __device__ virtual float GetActionDE(int action_num){
	// 	printf("Error! GetActionDE not implimented in child class\n");
	// 	return 0;
	// } 

	// //the potential energy; only really makes sense for Potts models.  Important only for my particular research purposes.
	// __host__ __device__ virtual float GetActionPE(int action_num){
	// 	printf("Error! GetActionPE not implimented in child class\n");
	// 	return 0;
	// } 

	// //changes internal state to reflect the annealing step that was taken
	// __host__ __device__ virtual void TakeAction(int action_num){
	// 	printf("Error! TakeAction not implimented in child class\n");
	// }



};