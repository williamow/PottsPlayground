import setuptools
import numpy
import shutil

#if NVCC is found, build with GPU support, else just build for CPU:
if shutil.which("nvcc") is not None:

    from setuptools_cuda_cpp import CUDAExtension, BuildExtension, fix_dll #builds using NVCC

    build={'build_ext': BuildExtension}
    ext = CUDAExtension(
            name="PottsPlayground.Annealing",
            sources=(
                "PottsPlayground/Solvers/Annealing.cpp",
                "PottsPlayground/Solvers/CpuCore.cpp",

                "PottsPlayground/Solvers/GpuCore.cu"
                ),
            include_dirs=[numpy.get_include()],
            libraries=["cudart"],

            extra_compile_args={"cxx": ["-std=c++17"], "nvcc": ["-std=c++17"]},
        )


else:
    build={}
    ext = setuptools.Extension(
            name="PottsPlayground.Annealing",
            sources=[
                "PottsPlayground/Solvers/Annealing.cpp",
                "PottsPlayground/Solvers/CpuCore.cpp",

                "PottsPlayground/Solvers/GpuCoreAlt.cpp"
                ],
            include_dirs=[numpy.get_include()],

            extra_compile_args=["-std=c++17"],
        )

setuptools.setup(
    name="PottsPlayground",
    # version=0.1,
    packages=['PottsPlayground', 'PottsPlayground.Tasks'],
    package_dir = {"PottsPlayground": "PottsPlayground", "PottsPlayground.Tasks": "PottsPlayground/Tasks"},
    ext_modules=[ext],
    cmdclass=build,
)

#good references for how to use setuptools:
# https://setuptools.pypa.io/en/latest/userguide/index.html
# https://packaging.python.org/en/latest/guides/modernize-setup-py-project/#modernize-setup-py-project