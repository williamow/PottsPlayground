.. PottsPlayground documentation master file, created by
   sphinx-quickstart on Mon Dec  2 13:57:35 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Fast Potts Models in Python
===========================

PottsPlayground is a Python package for constructing and simulating combinatorial optimization problems represented as Potts Models.  It's purpose is twofold: providing an easily readable interface for constructing Potts models, and providing a fast C++/CUDA backend for annealing or sampling from models.

Several example combinatorial problems are built-in, ranging from simple graph coloring to logic element placement for Ice40 FPGAs.  There are also methods for converting any Potts model into an Ising model, and for creating minor-embedded representations of Ising models.


.. toctree::
   :name: "Potts Playground"

   gettingstarted
   basetask
   tasks
   taskmodifiers
   annealing
   Schedules
   kernels