#nvcc version of building a python package
from setuptools import setup, Extension
from setuptools_cuda_cpp import CUDAExtension, BuildExtension, fix_dll
import os
import numpy

# os.environ["CUDAHOME"] = #don't seem to need this on my installation
# os.environ["CC"] = "g++-9" #can specify specific compiler version if needed
# os.environ["CXX"] = "g++-9"

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

            extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]},
        )

setup(
    name = 'Annealing',
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension},

)
