
#include "matrix_utils.h"

typedef __m128d scalar_t;
typedef __m128d complex_t;

#define TEST_MUL 1
#define TEST_EXP 0
#define TEST_ADD_HERM 0

int main(){
	complex_t zero = to_complex(0.0, 0.0);
	complex_t one = to_complex(1.0, 0.0);

  uint32_t* nnz_by_row;
  uint32_t** nnz_col_locations;
  complex_t** nnz_vals;

  uint32_t num_nnz = 2;	
  uint32_t num_rows = 2;
	uint32_t max_nnz = 1;

	nnz_vals = new complex_t*[num_nnz];
	nnz_by_row = new uint32_t[num_rows];
	nnz_col_locations = new uint32_t*[num_rows];
  
  for(uint32_t i=0; i<num_rows; i++)
  {
    nnz_vals[i] = new complex_t[max_nnz];
    nnz_col_locations[i] = new uint32_t[max_nnz];
  }

  for (uint32_t i=0; i<num_rows; i++)
  {
    for(uint32_t j=0; j<max_nnz; j++)
    {
    	nnz_col_locations[i][j] = 0;
      nnz_vals[i][j] = to_complex(0.0, 0.0);
    }
  }
  
  nnz_by_row[0] = 1;
  nnz_by_row[1] = 1;
  
	nnz_col_locations[0][0] = 0;
	nnz_col_locations[1][0] = 1;

  nnz_vals[0][0] = to_complex(0.52, 0.0);
  nnz_vals[1][0] = to_complex(0.52, 0.0);
  
  printf("About to pass to constructor\n");
//ComplexMatrix(uint32_t rows, uint32_t max_nnz, uint32_t* nnz_by_row,  complex_t** nnz_vals, uint32_t** nnz_col_locations)
  ComplexMatrix test_mtx(num_rows, max_nnz, nnz_by_row, nnz_vals, nnz_col_locations);
  test_mtx.print_compressed_storage_full();

  ComplexMatrix dst(num_rows, num_rows);
  test_mtx.exp_ham_sparse(dst, 1.0, 1e-25, false);
  
  
    
}
