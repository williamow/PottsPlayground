#nvcc version of building a python package
from setuptools import setup, Extension
from setuptools_cuda_cpp import CUDAExtension, BuildExtension, fix_dll
import os
import numpy

# os.environ["CUDAHOME"] = #don't seem to need this on my installation
os.environ["CC"] = "g++-9"
os.environ["CXX"] = "g++-9"

cuda_ext = CUDAExtension(
            name="Annealing",
            sources=(
                "PottsJitAnnealable.cu",
                "PottsPrecomputeAnnealable.cu",
                "TspAnnealable.cu",
                "Annealing.cu",
                ),
            include_dirs=[numpy.get_include()],
            libraries=["cudart"],

            # rdc (redistributable device code) is needed since "Annealable" __device__ object functions are
            # compiled separately from the __global__ kernel functions that use them.
            # if all code was compiled as a single source using includes, rdc would not be needed.
            extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]},
            # dlink=True
            # library_dirs = ["./"]extra_postargs=['nvcc_dlink'] #need extra nvcc link step for rdc code
        )

setup(
    name = 'Annealing',
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension},

)
