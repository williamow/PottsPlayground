import setuptools
import numpy
import shutil

with open("PottsPlayground/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            # Extract the version string (e.g., "__version__ = '0.1.0'")
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

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

            extra_compile_args={"cxx": ["-std=c++11"], "nvcc": ["-std=c++11"]},
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

            extra_compile_args=["-std=c++11"]#, "/std:c++11"], #first one is for linuxland, second for stupid MSVC
        )

setuptools.setup(
    name="PottsPlayground",
    version=version,
    packages=['PottsPlayground', 'PottsPlayground.Test', 'PottsPlayground.Tasks'],
    package_dir = { "PottsPlayground":          "PottsPlayground",
                    "PottsPlayground.Test":     "PottsPlayground/Test",
                    "PottsPlayground.Tasks":    "PottsPlayground/Tasks"},
    install_requires = ["matplotlib>=0.0.0",
                        "networkx>=2.0",
                        "numpy>=1.17.0",
                        #"blifparser>=0.0.0",
                        "minorminer>=0.0.0",
                        "dimod>=0.0.0",
                        "dwave-system>=0.0.0"
                        ],
    ext_modules=[ext],
    cmdclass=build,
)

#good references for how to use setuptools:
# https://setuptools.pypa.io/en/latest/userguide/index.html
# https://packaging.python.org/en/latest/guides/modernize-setup-py-project/#modernize-setup-py-project