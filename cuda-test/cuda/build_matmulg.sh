nvcc   -rdc=true --compiler-options -fPIC -c -o temp.o matmul.cu
nvcc -dlink --compiler-options -fPIC -o matmul.o temp.o -lcudart
rm -f libmmg.a
ar cru libmmg.a matmul.o temp.o
ranlib libmmg.a