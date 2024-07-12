#include path for numpy can be found by running numpy.get_include() within the target python environment
export Lnumpy=/home/williamow/.local/lib/python3.10/site-packages/numpy/core/include
#python include path obtained by python3 -c "import distutils.sysconfig as sc; print(sc.get_python_inc())"
export Lpython=/usr/include/python3.10

echo "Building ColoringRefs"
g++ -fPIC -c -o a.o ColoringRefs.cpp -I $Lpython

#bundle library:
echo "Assembling Library"
rm -f libTaskSolvers.a
ar cru libTaskSolvers.a a.o
ranlib libTaskSolvers.a

rm -f a.o

#-rdc= Re-Distributable Code
#-fPIC is a --compiler-option sent to the g++ side of things, needed so that python can dynamically link the library correctly
#-I $Lpython adds Lpython to the include search path.  Not sure why I didn't need to also do -I $Lnumpy...

