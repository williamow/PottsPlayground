export Lnumpy=/home/williamow/.local/lib/python3.10/site-packages/numpy/core/include
#python include path obtained by python3 -c "import distutils.sysconfig as sc; print(sc.get_python_inc())"
export Lpython=/usr/include/python3.10

# export compile_flags=$(

#suggested compile-time flags can be given by running:
python3-config --cflags

#and suggested link-time flags:
python3-config --ldflags --embed

#but these cannot be taken verbatim, since nvcc does not have all the same features, options, and behaviors that gcc has.


nvcc -g -G Annealing.cu -o a1.o -I/usr/include/python3.10 -L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lpython3.10 -lcrypt -ldl  -lm -lm

# -c -o a1.o  Annealing.cu   
# nvcc -dlink --compiler-options -fPIC -o a2.o a1.o -lcudart

# nvcc -run Annealing.cu -I $Lpython