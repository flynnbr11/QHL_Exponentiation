#include "matrix_utils.h"

int main() {

  complex_t complex_zero = to_complex(0.0, 0.0);	
  complex_t complex_one = to_complex(1.0, 0.0);	
	complex_t complex_i = to_complex(0.0, 1.0);
	complex_t complex_minus_i = to_complex(0.0, -1.0);


  complex_t conj = to_complex(1.0, -1.0);
	int size =4;
	
	complex_t mtx_data[size*size];
	for(uint32_t i=0; i<size; i++)
	{	
		for(uint32_t j=0; j<size; j++)
		{
			mtx_data[i*size + j] = complex_zero;		
		}
	}
	if(size==4){
		mtx_data[1] = complex_minus_i;
		mtx_data[4] = complex_i;
		mtx_data[11] = complex_i;
		mtx_data[14] = complex_minus_i;
	}
	else if (size==2){
		mtx_data[1] = complex_one;
		mtx_data[2] = complex_one;
	}
	
	
	ComplexMatrix ham(size, size, mtx_data);
	ham.debug_print();
	
	ComplexMatrix dest(size, size);
	double precision = 1e-12;
	ham.cos_plus_i_sin(dest, precision);
//	ham.expm_special(dest, precision);

	/*
	ComplexMatrix destination_expm_special(size, size);
	ComplexMatrix destination_cos_sin(size, size);

	ham.make_identity();
	ham.expm_special(destination_expm_special, 1e-15);
	ham.cos_plus_i_sin(destination_cos_sin, 1e-15);
	*/
	
	
	/*
	

	printf("Before Cos + i Sin:\n");
	ham.debug_print();
	printf("---- ----");
	printf("After Cos + i Sin:\n");
	destination_cos_sin.debug_print();
	printf("---- ----");
	printf("After expm special: \n");
	destination_expm_special.debug_print();
	*/

}
