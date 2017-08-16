#include "matrix_utils.h"

int main() {

  complex_t complex_zero = to_complex(0.0, 0.0);	
  complex_t complex_one = to_complex(1.0, 0.0);	
	complex_t complex_i = to_complex(0.0, 1.0);
	complex_t complex_minus_i = to_complex(0.0, -1.0);
  complex_t conj = to_complex(1.0, -1.0);
	int size =2;
	
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
		double pauli_x = 1.0;
		double pauli_y = 0.0;
		mtx_data[1] = to_complex(pauli_x, -1.0*pauli_y);
		mtx_data[2] = to_complex(pauli_x, pauli_y);
	}
	double precision = 1e-15;
	double time = 1.0;

	ComplexMatrix ham(size, size, mtx_data);
	printf("Input Matrix: \n");
	ham.debug_print();
	ComplexMatrix dest(size, size);
	
	//*
	ham.expm_minus_i_h_t(dest, time, precision);
	printf("Output: \n");
	dest.debug_print();
	//*/
	/*	
	double normalisation_scalar = ham.normalise_matrix();
	printf("After normalisation with scalar %lf:\n", normalisation_scalar);
	ham.debug_print();
	//*/
	

}
