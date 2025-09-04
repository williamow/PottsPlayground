===============
Running a Model
===============

Annealing API
=============
There is a single C++/CUDA extension that performs annealing.  However there are a large number of options, which are described in greater detail below.

.. py:function:: PottsPlayground.Anneal(task, PwlTemp, **kwargs)

   Anneals/samples/computes a Potts model and returns a dictionary of results.

   Required parameters:

   :param task: A PottsPlayground task object containing a complete Potts Model description.
   :param PwlTemp: An array describing the annealing schedule, i.e. the annealing temperature as a piece-wise linear function of iterations.
   :type PwlTemp: 2-D numpy float.  See Schedules page for format.

   Parameters for selecting the computational method:

   :param model: What model to anneal.  See "Models" section. Default: "Potts"
   :type model: str
   :param device: Which physical device to run the annealing on.  Either "CPU" or "GPU".  Defaults to CPU if the project was not compiled with GPU support, or if no suitable GPU was found on import.
   :type device: str
   :param algo: Selects between sampling algorithm variants.  Available selections depends on the annealing device.
   :type algo: str

   Additional run-time controls:

   :param nReplicates: Number of Potts Model instances to compute in parallel.
   :type nReplicates: int
   :param nWorkers: On a CPU, the number of CPU threads to use; on a GPU, the number of GPU threads that work together collectively on a single replicate. On a GPU, depending on the algorithm, this can alter results.
   :type nWorkers: int
   :param IncludedReports: Str containing letter flags for which types of information are returned by the function. See "Reports" section.
   :param IC: optional Initial Condition for the annealing run.  Either a 1-D Numpy int array containing a single IC for all replicates, or a 2-D Numpy array of ints (#Replicates X #Spins) indicating a distinct initial condition for each replicate.
   :param nReports: How many times over the course of the annealing run annealing should be stopped and vitals checked and reported. If -1, Reports are taken at the points defined in PwlTemp.
   :type nReports: int

   Algorithm-specific parameters (see algorithms section):

   :type nOptions: int (OptionsActions algorithm)
   :type nActions: int (OptionsActions algorithm)

   :return: A collection of energies, states, and more that describe the outcome of the annealing.
   :rtype: dict


Models
======
Annealing can be done with one of several computational "models", which is usually the Potts Model.  However Ising model and a natural Traveling Salesman Model are also included for comparison purposes.  Furthermore two Potts model variants are available; they function identically, but depending on other factors one or the other will execute faster.

model="Potts"
-------------
The normal Potts backend computes the DE of changing a spin only when the spin change is actually proposed in the annealing algorithm. This is the normal mode of computation found across simuated annealing implimentations in the software world.

model="PottsPrecompute"
-----------------------
The Potts Precompute backend always knows the DE of every single possible spin change.  The DE values are kept current by an update algorithm that calculates DDE values after every spin change that occurs; this is more efficient when many possible DE values are queried between actual spin changes.

model="Ising"
-------------
When a Potts model has q=2 for all spins, and the weight kernels are specified in a binary-quadratic format, the Ising backend provides a more streamlined but functionally equivalent computation.

model="IsingPrecompute"
-----------------------
Incrimentally updates DE values in the Ising model, the same way the PottsPrecompute model incrimentally keeps track of DE in the Potts model.

model="TSP"
-----------
The TSP background is for use with the traveling salesman task only, and defines a completely different model of computation where the order of cities is strictly enforced and only city swaps are allowed.  This is the standard annealing format for the traveling salesman problem outside of the realm of physics-inspired computing.

Parallelism
===========

Several types of parallelism exist when a model is annealed.  Two main types are mechanical and concern the acceleration of an annealing run using multithreading.  Additional types of parallelism depend on the sampling algorithm - see Algorithms section.



Replicates (**nReplicates**)
----------------------------
Multiple copies of a Potts or other model can be executed in parallel.  This is particularly helpful for sampling, since the different copies will yield independent states suitable for taking averages, etc.  For many problem-solving annealing runs however, performance is often limited by the mixing/settling time of a single replicate, and a large population of replicates provides rapidly diminishing returns.

Multithreading (**nWorkers**)
-----------------------------
This parameter has different effects for GPU and CPU operation.  On a CPU, each model replicate is contained within a single thread, and nWorkers defines how many threads can run at once - usually, the number of cores in the system.

For a GPU, threads can more easily work together to speed up computation on a single replicate.  In this case nWorkers specifies how many GPU threads should work cooperatively on each replicate; this value should be chosen based on the level of parallelism specified by nOptions and nActions.  If there is only one option and one action, multiple cooperating GPU threads won't provide much of a speed-up; however if there are many, 32 workers is a good choice (the number of threads in an Nvidia GPU warp; something to read about elsewhere).  A GPU call will automatically use as many of the available GPU cores as possible, i.e. 1024 cores if there are 32 replicates each with 32 workers.

Algorithmic
-----------
The most common type of algorithmic parallelism is updating multiple Potts/Ising spins at once.  Restricted Boltzmann machines are in fact restricted precisiely so that whole sets of Ising spins can be updated at the same time, without violating Boltzmann statistics, in a way that is computationally fast.  However, other forms of parallelism also exist; see below for more details on how different algorithms provide parallelism in different ways.


Algorithms
==========

Several sampling algorithms are available, selected by the **algo** parameter.

algo="Simple" (CPU only)
------------------------
This is the tradtional simulated annealing algorithm, where state changes are proposed, the change in cost dE is evaluated, and then the change is accepted if dE is negative or is accepted with probability exp(-dE/T) otherwise.

algo="OptionsActions"
---------------------
At each step in the "OptionsActions" algorithm, **nOptions** possilbe spin changes are considered at a time, and up to **nActions** spin updates actually happen.  When **nOptions**=1 and **nActions**=1, it behaves the same as the Simple algorithm.  When **nOptions** is large and **nActions**=1, it behaves similarly to the BirdsEye algorithm, but is less efficient.  By enabling parallel spin updates with **nActions**>1 and varying **nOptions**, this algorithm can trade off computational efficiency and algorithmic efficiency.


algo="KMC" (CPU only)
---------------------
Uses the kinetic Monte Carlo algorithm.  At each computational step, all possible spin changes are considered simultaneously.  In order to do this efficiently in software, the algorithm maintains a binary tree of spin update probabilities to enable searching in logarithmic time, and sparse recomputation of probability changes after each spin flip.  For sparse recomputation to work efficiently, either the "PottsPrecompute" or "IsingPrecompute" models need to be used.


Reports
=======
The Annealing function returns a dictionary of Reports.  Each report concerns a particular aspect or measure of the annealing run, formatted as a Numpy array.  The first dimension is always equal to the number or reports; some reports are only a scalar value at each sampling point, while others which may be a vector or matrix at each sampling point return Numpy matrices with two or three dimensions respectively.

There are two ways the reporting schedule can be defined.  If **nReports** is given, the iterations of the annealing run will be split into **nReports** segments, and the performance/behavior of the annealing run will be recorded at the end of each segment.  If nReports is not defined, the reports are made at the end of each segment defined in the piece-wise linear annealing schedule definition.

In this list, the letter corresponds to the flag that includes the report, and the name indentifier can be used to find the information in the returned dictionary.

**A - "MinEnergies" - [#Reports]** - The minimum energy acheived by any replicate at any iteration over the course of the annealing run.

**B - "AvgEnergies" - [#Reports]** - The average energy of all the replicates at the time each report is gathered.

**C - "AvgMinEnergies" - [#Reports]** - The minimum energies of each replicate from any iteration, averaged.

**D - "AllEnergies" - [#Reports, #Replicates]** - Individual energy of each replicate at the time each report is gathered.

**E - "AllStates" - [#Reports, #Replicates, #Spins]** - Every spin value from every replicate, at the time each report is gathered.

**F - "AllMinStates" - [#Reports, #Replicates, #Spins]** - Best (lowest energy) configurations found by each replicate by the time each report is gathered.

**G - "MinStates" - [#Reports, #Spins]** - Overall best (lowest energy) configuration, across all replicates, found by the time each report is gathered.  Corresponds to the **MinEnergies** report.

**H - "DwellTimes" - [#Reports, #Replicates]** - Niche use, for correcting statistical biases when running fully-parallel hypothesis annealing.  Since the annealing core selects decisions on a first-to-fire principle, samples must be weighted by their dwell time to correctly match a Boltzmann distribution.

**I - "Iter" - [#Reports]** - For convienience, the number of iterations at each report.

**J - "Temp" - [#Reports]** - The annealing temperature at each report.

**L - "RealFlips" - [#Reports]** - Execution iterations defines oppurtunities for spin flips, but often none of the proposed spin updates are actualized.  This counts the actualized spin flips taken up until the time of each report.

**M - "AllMinEnergies" - [#Reports, #Replicates]** - The minimum energy of each replicate.