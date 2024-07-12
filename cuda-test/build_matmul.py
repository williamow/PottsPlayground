from setuptools import setup, Extension
import numpy
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

# if 'CUDA_PATH' in os.environ:
#    CUDA_PATH = os.environ['CUDA_PATH']
# else:
#    print("Could not find CUDA_PATH in environment variables. Defaulting to /usr/local/cuda!")
#    CUDA_PATH = "/usr/local/cuda"

# if not os.path.isdir(CUDA_PATH):
#    print("CUDA_PATH {} not found. Please update the CUDA_PATh variable and rerun".format(CUDA_PATH))
#    exit(0)


ext_modules = [
    Extension(
        'matmul',
        sources = ['matmul.cpp'],
        include_dirs=[numpy.get_include()],
        libraries=["mmg", "cudart"],
        library_dirs = ["./cuda"],
        extra_link_args = ['-fPIC'] #something needed to deal with linking issues?
    ),
]

setup(
    name = 'matmul',
    ext_modules = ext_modules
)