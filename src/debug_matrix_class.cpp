#include "matrix_utils.h"

int main(){

  complex_t conj = to_complex(1.0, -1.0);
	const int size =4;
	const int num = size*size;
	complex_t mtx_data[size][size]; 
	complex_t mtx_mul_data[size][size];
	complex_t mtx_data_ptr[size*size]; 
	complex_t mtx_mul_data_ptr[size*size];

  complex_t complex_zero = to_complex(0.0, 0.0);	
  complex_t complex_one = to_complex(1.0, 0.0);	

//*
	for (int i =0; i<size; i++){
		for (int j=0; j<size; j++){
			mtx_data[i][j] = complex_zero;
			mtx_mul_data[i][j] = complex_zero;
		}
	}
//*/
	 
	mtx_data[1][2] = to_complex(2.0, 2.0);
	mtx_data[2][1] = mtx_data[1][2] * conj;
	mtx_data[1][3] = to_complex(3.0, 3.0);
	mtx_data[3][1] = mtx_data[1][3] * conj;

	mtx_mul_data[1][2] = to_complex(2.0, 2.0);
	mtx_mul_data[2][1] = mtx_mul_data[1][2] * conj;
	mtx_mul_data[1][3] = to_complex(3.0, 3.0);
	mtx_mul_data[3][1] = mtx_mul_data[1][3]*conj;
//	mtx_mul_data[9] = to_complex(7.0, 1.0);
	


	for (int i =0; i<size; i++){
		for (int j=0; j<size; j++){
			mtx_data_ptr[i*size+j] = mtx_data[i][j];
			mtx_mul_data_ptr[i*size+j] = mtx_mul_data[i][j];
		}
	}


	ComplexMatrix test_matrix(size, size, mtx_data_ptr);
	ComplexMatrix mult_mtx(size, size, mtx_mul_data_ptr);
  ComplexMatrix dest(size, size);
	
	printf("Multiplying This : \n");
	test_matrix.debug_print();
	test_matrix.compress_matrix_storage();
	printf("By RHS : \n");
	mult_mtx.debug_print();
//	test_matrix.print_compressed_storage();
	
/*	
	printf("Before mult, test_matrix: \n");
	test_matrix.debug_print();
	printf("Before mult, mult_mtx: \n");
	mult_mtx.debug_print();
	
	printf("Before mult, dest: \n");
	dest.debug_print();
*/	
	test_matrix.mul_hermitian(mult_mtx, dest); 
	printf("After mult: \n");
	dest.debug_print();

/*
	dest.compress_matrix_storage();
	printf("After compression: \n");
	dest.print_compressed_storage();
*/
}
