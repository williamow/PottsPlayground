//a little class for determining the annealing temperature from a piece-wise linear specification
//the class uses hard-coded storage of the piece-wise linear specification, only allowing up to 31 segments.
//this way, I don't have to worry about memory management between the host and the gpu :)

#ifndef PieceWiseLinearHeader
#define PieceWiseLinearHeader

#include "NumCuda.h"

class PieceWiseLinear{
public:

	float f[128];
	long x[128];
	int nSegments;

	float slope;
	float offset;

	long LastDiscontinuity;
	long NextDiscontinuity;

    // ================================================================================= CONSTRUCTORS, DESTRUCTORS, ETC
    __h__ PieceWiseLinear(NumCuda<float> &source){
        //initialize from a NumCuda array:
        nSegments = source.dims[1];
        if (nSegments > 128) printf("Error: PieceWiseLinear class can only handle a maximum of 128 points, the source has %i points\n", nSegments);
        for (int i = 0; i<nSegments; i++){
        	f[i] = source(0, i);
        	x[i] = long(source(1, i));
        }
        LastDiscontinuity = 1; //an initial setting that will force a segment identifcation at the first interpolation request
        NextDiscontinuity = 0;
    }

    __h__ __d__ PieceWiseLinear(){}

    __h__ __d__ float interp(long iter){
    	//interpolation, designed to be efficient when the iter repeatadly draws from the same line segment before switching 
    	if (iter < LastDiscontinuity || iter > NextDiscontinuity){
    		//we need to find the correct operating segment
    		int i;
    		for (i = 1; i < nSegments; i++)
    			if (x[i] > iter) break;
    		NextDiscontinuity = x[i];
    		LastDiscontinuity = x[i-1];
    		slope = (f[i]-f[i-1])/(NextDiscontinuity-LastDiscontinuity);
    		offset = f[i-1] - slope*LastDiscontinuity;
    	}
    	return iter*slope + offset;
    }

};


#endif