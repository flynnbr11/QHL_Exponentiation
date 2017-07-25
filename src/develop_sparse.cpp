
#include "matrix_utils.h"

typedef __m128d scalar_t;
typedef __m128d complex_t;
inline complex_t to_complex(double r, double i) { return _mm_set_pd(i, r); }


int main(){
	complex_t zero = to_complex(0.0, 0.0);
	complex_t one = to_complex(1.0, 0.0);

	complex_t* vals;

	uint32_t r=4;
	uint32_t c=4;
	vals = new complex_t[r*c];	
	
	for (uint32_t i=0; i<r; i++)
	{
		for (uint32_t j=0; j<c; j++)
		{
			vals[i*r + j] = zero;
		}
	}

	ComplexMatrix mtx(r,c, vals);
	//mtx.make_identity();
	delete[] vals;
	return 0;
}


