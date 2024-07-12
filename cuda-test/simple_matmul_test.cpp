#include <stdio.h>

double * myVectorAdd(double * h_A, double * h_B, int numElements);

int main(){

	double A [4] = {1,1,1,1};
	double B [2] = {1,2};
	int n = 2;

	double * output = myVectorAdd(A, B, n);

	printf("Output matrix values are: %.2f, %.2f\n", output[0], output[1]);


	return 0;
}