from setuptools import setup, Extension
import numpy
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

ext_modules = [
    Extension(
        'AllSolvers',
        sources = ['AllSolvers.cpp'],
        # include_dirs=[numpy.get_include()], #I think not needed anymore since code meat was moved to statically linked libraries?
        libraries=["PottsSolvers", "TaskSolvers", "cudart"],
        library_dirs = ["./PottsSolvers", "./TaskSolvers"]
    ),
]

setup(
    name = 'AllSolvers',
    ext_modules = ext_modules
)