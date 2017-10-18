#include "matrix_utils.h"

typedef __m128d scalar_t;
typedef __m128d complex_t;

#define TEST_MUL 0
#define TEST_ADD_HERM 0
#define TEST_EXP 1

#define TEST_MTX 1

int main()
{
	complex_t zero = to_complex(0.0, 0.0);
	complex_t one = to_complex(1.0, 0.0);
	complex_t two = to_complex(2.0, 0.0);

  uint32_t* nnz_by_row;
  uint32_t** nnz_col_locations;
  complex_t** nnz_vals;

  uint32_t num_rows;
  uint32_t num_nnz;	
  uint32_t max_nnz;
  
  
  if(TEST_MTX == 1)
  {
    num_rows = 2;
    num_nnz = 2;	
	  max_nnz = 1;

  }
  else if(TEST_MTX==2)
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
  
  if(TEST_MTX==1)
  {

    nnz_by_row[0] = 1;
    nnz_by_row[1] = 1;    
	  nnz_col_locations[0][0] = 1;
	  nnz_col_locations[1][0] = 0;
    nnz_vals[0][0] = to_complex(0.52, -0.08);
    nnz_vals[1][0] = to_complex(0.52, 0.08);

  }
  
  if(TEST_MTX==2)
  {

    nnz_by_row[0] = 1;
    nnz_by_row[1] = 1;    
    nnz_by_row[2] = 1;
    nnz_by_row[3] = 1;    
	  nnz_col_locations[0][0] = 3;
	  nnz_col_locations[1][0] = 2;
	  nnz_col_locations[2][0] = 1;
	  nnz_col_locations[3][0] = 0;
    nnz_vals[0][0] = to_complex(0.52, -0.08);
    nnz_vals[1][0] = to_complex(0.52, 0.08);
    nnz_vals[2][0] = to_complex(0.52, -0.08);
    nnz_vals[3][0] = to_complex(0.52, 0.08);

  }


  ComplexMatrix test_mtx(num_rows, max_nnz, nnz_vals, nnz_by_row, nnz_col_locations);

  ComplexMatrix mtx_two(num_rows, max_nnz, nnz_vals, nnz_by_row, nnz_col_locations);

  ComplexMatrix dst(num_rows, num_rows);
  
  
  if(TEST_ADD_HERM)
  {

    printf("\n\nAdding : LHS \n");
    mtx_two.print_compressed_storage_full();
    printf("\nAdding : RHS \n");
    test_mtx.print_compressed_storage_full();
    printf("\n\n");
    mtx_two.add_complex_scaled_hermitian_sparse(test_mtx, one);    

    printf("\n \n \nSUM: \n");
    mtx_two.print_compressed_storage_full();
  }
  
  
  if(TEST_MUL)
  {
    test_mtx.sparse_hermitian_mult(mtx_two, dst);

    printf("\n \n \nLHS: \n");
    test_mtx.print_compressed_storage_full();

    printf("\n \n \nRHS: \n");
    mtx_two.print_compressed_storage_full();

    printf("\n \n \nProduct: \n");
    dst.print_compressed_storage_full();
  }  
  if(TEST_EXP)
  {
    ComplexMatrix sparse_dst(num_rows, num_rows); 
    
    test_mtx.exp_ham(dst, 1.0, 1e-25, false);
    test_mtx.exp_ham_sparse(sparse_dst, 1.0, 1e-25, false);

    printf("\n\n---- ---- Input ---- ---- \n");
    test_mtx.print_compressed_storage_full();

    printf("\n \nSPARSE: \n");
    sparse_dst.print_compressed_storage_full();


    printf("\n \n \nACTUAL: \n");
    dst.print_compressed_storage_full();

  }
  
	return 0;

}

