int nThrds = commspace.dims[1]; //threads per replicate
int nReplicates = commspace.dims[0];
int replicate_index = ThrdIndx/nThrds;
int Thrd = ThrdIndx%nThrds; //thread index within each replicate
if (replicate_index + 1 > nReplicates) return;

Annealable* task = &taskIn;

//create local references to the state vectors used in this particular thread:
int *MiWrk = &working_states(replicate_index, 0);
int *MiBest = &best_states(replicate_index, 0);

task->SetIdentity(Thrd, nThrds, replicate_index, MiWrk, MiBest);
task->BeginEpoch(MinIter);

int iter_inc = 1;
if (TakeAllActions) iter_inc = nThrds;

for (int iter = MinIter; iter < MaxIter; iter+=iter_inc){
    if (*TaskDone > 0) break;
    SYNC_REPLICATE();
    //each thread selects the best action within its subset:
    float T = PwlTemp.interp(iter);

    int action = -1; //start with default Null action, with dE=0.
    float BestPriority = -log2(RngUniform()); //for keeping track of which action will be best
    for (int i=0; i<OptsPerThrd;i++){
        int possible_action = RngInteger()%task->NumActions;
        float possible_dE = task->GetActionDE(possible_action);

        //switched signs in order to convert divisions to multiplications:
        float NhppRate = exp2(possible_dE/T); //exp2(-possible_dE/T); 
        float NewPriority = -log2(RngUniform())*NhppRate; ///NhppRate;
        // float NewPriority = possible_dE; //short circuit the NHPP

        if (NewPriority < BestPriority){
            BestPriority = NewPriority;
            action = possible_action;
        }
    }
    commspace(replicate_index, Thrd, 0) = BestPriority;
    commspace(replicate_index, Thrd, 1) = action;

    //synchronize threads, so that all threads can see the action proposals from the other threads.
    //this may sync with other threads that are not working cooperatively, too;
    //while that might be undesirable for performance, it is functionally okay.
    SYNC_REPLICATE();

    //all threads simultaneously decide which actions to take on their annealable object, although this is redundant.
    //They should all be doing exactly the same thing, if they are not, that would be bad.
    if (TakeAllActions){
        //all threads take the same nThrds actions
        for (int i=0;i<nThrds;i++){
            action = commspace(replicate_index, i, 1);
            if (action >= 0)
                task->TakeAction_tic(action);
            SYNC_REPLICATE();
            task->TakeAction_toc(action);
            SYNC_REPLICATE();
        }
    } else {
        //finds the single best action from all threads, and takes that action.
        for (int i=0;i<nThrds;i++){
            if (commspace(replicate_index, i, 0) <= BestPriority){
                BestPriority = commspace(replicate_index, i, 0);
                action = commspace(replicate_index, i, 1);
            }
        }
        if (action >= 0)
            task->TakeAction_tic(action);
        SYNC_REPLICATE();
        task->TakeAction_toc(action);
    }
    if (task->lowest_e < e_th) *TaskDone = 1; 
}

working_energies(replicate_index) = task->current_e;
best_energies(replicate_index) = task->lowest_e;
task->FinishEpoch();