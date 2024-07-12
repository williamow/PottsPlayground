#include path for numpy can be found by running numpy.get_include() within the target python environment
export Lnumpy=/home/williamow/.local/lib/python3.10/site-packages/numpy/core/include
#python include path obtained by python3 -c "import distutils.sysconfig as sc; print(sc.get_python_inc())"
export Lpython=/usr/include/python3.10

echo "Building GpuFullWeights"
nvcc -rdc=true --compiler-options -fPIC -c -o a1.o   GpuFullWeights.cu   -I $Lpython
nvcc -dlink --compiler-options -fPIC -o a2.o a1.o -lcudart

echo "Building GpuKernelWeights"
nvcc -rdc=true --compiler-options -fPIC -c -o b1.o GpuKernelWeights.cu -I $Lpython
nvcc -dlink --compiler-options -fPIC -o b2.o b1.o -lcudart

echo "Building CpuSingleThread"
g++ -fPIC -c -o c.o CpuSingleThread.cpp -I $Lpython

# echo "Building NumCuda"
# g++ -fPIC -c -o d.o NumCuda.cpp -I $Lpython


#bundle library:
echo "Assembling Library"
rm -f libPottsSolvers.a
ar cru libPottsSolvers.a a1.o a2.o b1.o b2.o c.o
ranlib libPottsSolvers.a

rm -f a1.o
rm -f a2.o
rm -f b1.o
rm -f b2.o
rm -f c.o 
# rm -f d.o

#-rdc= Re-Distributable Code
#-fPIC is a --compiler-option sent to the g++ side of things, needed so that python can dynamically link the library correctly
#-I $Lpython adds Lpython to the include search path.  Not sure why I didn't need to also do -I $Lnumpy...

