import setuptools
from setuptools_cuda_cpp import CUDAExtension, BuildExtension, fix_dll #builds using NVCC
import numpy

cuda_ext = CUDAExtension(
            name="PottsPlayground.Annealing",
            sources=(
                "Solvers/PottsJitAnnealable.cu",
                "Solvers/IsingAnnealable.cu",
                "Solvers/PottsPrecomputeAnnealable.cu",
                "Solvers/TspAnnealable.cu",
                "Solvers/Annealing.cu",

                #these ones compile only by include, but I include them so that changes trigger re-compiling:
                "Solvers/AnnealingCore.cu"
                ),
            include_dirs=[numpy.get_include()],
            libraries=["cudart"],

            extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]},
        )

setuptools.setup(
    name="PottsPlayground",
    # version=0.1,
    packages=['PottsPlayground', 'PottsPlayground.Tasks'],
    package_dir = {"PottsPlayground": "init", "PottsPlayground.Tasks": "Tasks"},
    ext_modules=[cuda_ext],
    cmdclass={'build_ext': BuildExtension},
)

#good references for how to use setuptools:
# https://setuptools.pypa.io/en/latest/userguide/index.html
# https://packaging.python.org/en/latest/guides/modernize-setup-py-project/#modernize-setup-py-project