#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
//#include "matrix_utils.h"
typedef __m128d scalar_t;
typedef __m128d complex_t;


complext_t *values;

uint32_t r=4;
uint32_t c=4;
for (uint32_t i=0; i<r; i++)
{
	for (uint32_t j=0; j<r; j++)
	{
		values[i*r + j] = 1;
	}
}



//ComplexMatrix mtx(r,c, values);
//mtx.make_identity();



