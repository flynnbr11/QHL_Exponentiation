#include "matrix_utils.h"

typedef __m128d scalar_t;
typedef __m128d complex_t;

#define TEST_MUL 0
#define TEST_ADD_HERM 0
#define TEST_EXP 1
#define TEST_MUL_SUM 0
#define TEST_EXP_HAM_OLD 0

#define TEST_MTX 1

int main()
{
	complex_t zero = to_complex(0.0, 0.0);
	complex_t one = to_complex(1.0, 0.0);
	complex_t two = to_complex(2.0, 0.0);
  complex_t conj = to_complex(1.0, 1.0);
  uint32_t* nnz_by_row;
  uint32_t** nnz_col_locations;
  complex_t** nnz_vals;

  uint32_t num_rows;
  uint32_t num_nnz;	
  uint32_t max_nnz;
  
  
  if(TEST_MTX == 1)
  {
    num_rows = 4;
    num_nnz = 4;	
	  max_nnz = 2;

  }

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
  complex_t alpha = to_complex(0.44, 0.33);
  complex_t alpha_conj = to_complex(0.44, -0.33);
  if(TEST_MTX==1)
  {

    nnz_by_row[0] = 1;
    nnz_by_row[1] = 2;    
    nnz_by_row[2] = 2;    
    nnz_by_row[3] = 1;    
	  nnz_col_locations[0][0] = 0;
	  nnz_col_locations[1][0] = 1;
	  nnz_col_locations[1][1] = 2;
	  nnz_col_locations[2][0] = 1;
	  nnz_col_locations[2][1] = 2;
	  nnz_col_locations[3][0] = 3;
    nnz_vals[0][0] = to_complex(0.52, 0.0);
    nnz_vals[1][0] = to_complex(0.52, 0.0);
    nnz_vals[1][1] = alpha;
    nnz_vals[2][0] = alpha_conj;
    nnz_vals[2][1] = to_complex(0.52, 0.0);
    nnz_vals[3][0] = to_complex(0.52, 0.0);
  }
  
  ComplexMatrix test_mtx(num_rows, max_nnz, nnz_vals, nnz_by_row, nnz_col_locations);
  
  ComplexMatrix dst(num_rows, num_rows);

  test_mtx.exp_ham_sparse(dst, 1.0, 1e-25, false);
  printf("\n\n\n");
  printf("Input : \n");
  test_mtx.print_compressed_storage_full();  
  printf("\n\nResult : \n");
  dst.print_compressed_storage_full();  
  
  return 0;
}
