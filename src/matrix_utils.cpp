// Copyright 2017 University of Bristol
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include "matrix_utils.h"

#define VERBOSE 0
#define PRINT_LINE_DEBUG 1
#define SPARSE_METHOD 0
#define PRINT_MATRICES_MULT 0


void ComplexMatrix::mag_sqr(RealMatrix& dst) const
{
    const complex_t* psrc = get_row(0);
    double* pdst = dst.get_row(0);
    for (uint32_t i = 0; i < num_values; ++i)
        *pdst++ = ::mag_sqr(*psrc++);
}

void ComplexMatrix::make_identity()
{
    complex_t zero = to_complex(0.0, 0.0);
    complex_t one = to_complex(1.0, 0.0);
    for (uint32_t row = 0; row < num_rows; ++row)
    {
        complex_t* prow = get_row(row);
        for (uint32_t col = row; col < num_cols; ++col)
        {
            complex_t* pcol = get_row(col);
            if (row == col)
                prow[col] = pcol[row] = one;
            else
                prow[col] = pcol[row] = zero;
        }
    }
}


void ComplexMatrix::make_zero()
{
    complex_t zero = to_complex(0.0, 0.0);
    for (uint32_t i = 0; i < num_cols*num_rows; ++i)
    {
			values[i] = zero;
    }
}



void ComplexMatrix::print_compressed_storage() const
{
	printf("[print %d] Max nnz in any row : %u \n", __LINE__, max_nnz_in_a_row);

	for(uint32_t i=0; i<num_rows; i++){
		printf("[print %d] There are %u Nonzeros in row %u :\n", __LINE__, num_nonzeros_by_row[i], i);
		for(uint32_t j=0; j<num_nonzeros_by_row[i]; j++)
		{
		  printf("\t (i,j) = (%u, %u)", i, j);
			printf("\t Loc :%u, \t", nonzero_col_locations[i][j]);
			complex_t val = nonzero_values[i][j];
			printf("Val :\t  %.4e+%.4ei , \n", get_real(val), get_imag(val));
		}
	}
  printf("[print %d] End of print\n",__LINE__);
}


void ComplexMatrix::print_compressed_storage_full() const
{
  printf("[print compressed - full]\n");
	printf("[print %d] Max nnz in any row : %u \n", __LINE__, max_nnz_in_a_row);

	for(uint32_t i=0; i<num_rows; i++){
		printf("[print %d] There are %u Nonzeros in row %u :\n", __LINE__, num_nonzeros_by_row[i], i);
		for(uint32_t j=0; j<max_nnz_in_a_row; j++)
		{
		  printf("\t (i,j) = (%u, %u)", i, j);
			printf("\t Loc :%u, \t", nonzero_col_locations[i][j]);
			complex_t val = nonzero_values[i][j];
			printf("Val :\t  %.4e+%.4ei , \n", get_real(val), get_imag(val));
		}
	}
}


#define OPT_3 0 // opt 3 correctly exploits Symmetrical shape, but not sparsity. 
#define OPT_4 1 // opt 4 used for development of sparsity utility
#define mul_full 0


// dst = this * rhs
void ComplexMatrix::mul_hermitian(const ComplexMatrix& rhs, ComplexMatrix& dst) // changing const of rhs
{
    size_t size = num_rows;
    complex_t zero = to_complex(0.0, 0.0);
    complex_t conj = to_complex(1.0, -1.0);
		
#if OPT_4
		// printf("OPT 4 \n");
		dst.make_zero();
    for (uint32_t row = 0; row < size; ++row)
    {
			if(this->num_nonzeros_by_row[row] != 0)
			{
				complex_t* dst_row = dst.get_row(row);
				for (uint32_t col = row; col<size; ++col)
				{
					if (rhs.num_nonzeros_by_row[col] != 0) 
					{
						complex_t accum = zero;
			  		
						for (uint32_t j = 0; j < this -> max_nnz_in_a_row; j++)
						{
							uint32_t col_loc = this->nonzero_col_locations[row][j];
							if(col_loc != num_cols)
							{
								for (uint32_t k=0; k < rhs.max_nnz_in_a_row; k++)
								{
									if (col_loc == rhs.nonzero_col_locations[col][k])
									{
										accum = add(accum, mul(this->nonzero_values[row][j], rhs.nonzero_values[col][k]*conj ));					
									}					
								}
							}
						}
						dst_row[col] = accum;
						dst[col][row] = accum * conj; 
					}
					}
				} // end for (row) loop

		}

#elif OPT_3

		for (uint32_t row = 0; row < size; ++row)
		{
      // Here we make the assumption that the output will be symmetric across
      // the diagonal, so we only need to calculate half of it, and  then
      // write the other half to match.
      complex_t* dst_row = dst.get_row(row);
      const complex_t* src1 = get_row(row);
      for (uint32_t col = row; col < size; ++col)
      {
          const complex_t* src2 = rhs.get_row(col);
          complex_t accum = zero;
          for (uint32_t i = 0; i < size; ++i)
              accum = add(accum, mul(src1[i], src2[i] * conj));
          dst_row[col] = accum;
          dst[col][row] = accum * conj; // This * conj may belong on the previous line
      }
    } // end for (row) loop
#elif mul_full
		for (uint32_t row = 0; row < size; ++row)
		{
      complex_t* dst_row = dst.get_row(row);
			for (uint32_t col =0; col<size; ++col)
			{
				complex_t accum = zero;
				for (uint32_t i = 0; i < size; ++i)
				{
					accum = add(accum, mul((*this)[row][i], rhs[i][col]));
				}
				dst_row[col] = accum;
			}
		}
#endif // end if opt4, opt3 
}


void ComplexMatrix::sparse_hermitian_mult(const ComplexMatrix& rhs, ComplexMatrix& dst)
{
    bool print_mult = false;
    if(PRINT_MATRICES_MULT)
    {
      printf("[Sparse Mult] LHS: \n");
      this->print_compressed_storage_full();
      printf("[Sparse Mult] RHS: \n");
      rhs.print_compressed_storage_full();
    }
     
    // Multiplication of one sparse Hermitian matrix by another. 
    size_t size = this->num_rows;
    complex_t zero = to_complex(0.0, 0.0);
    complex_t conj = to_complex(1.0, -1.0);
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);
    uint32_t num_rows = this->num_rows;
    if(print_mult) ("[mult %d] num rows = %u \n", __LINE__, num_rows);
		//dst.make_zero();
    uint32_t num_elements = num_rows*num_rows;
		complex_t nonzero_values[num_elements];
    uint32_t rows[num_elements];
    uint32_t cols[num_elements];
    uint32_t row_full_upto[num_rows];
    uint32_t k = 0;
    uint32_t total_num_nnz = 0;    
  
    uint32_t temp_nnz_by_row[num_rows];
    for(uint32_t i=0; i<num_rows; i++)
    {
      temp_nnz_by_row[i] = 0;
    }
    
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

    if(print_mult) ("[mult %d] Start of mult, matrices: \n", __LINE__);
    if(print_mult) ("[mult %d] THIS\n", __LINE__ );
    this->print_compressed_storage_full();
    if(print_mult) ("[mult %d] RHS\n", __LINE__ );
    rhs.print_compressed_storage_full();
    for (uint32_t row = 0; row < size; ++row)
    {
      if(print_mult) ("[mult %d] row %d \n", __LINE__, row);
			if(this->num_nonzeros_by_row[row] != 0)
			{
//				complex_t* dst_row = dst.get_row(row);
				for (uint32_t col = row; col<size; ++col)
				{
          if(print_mult) ("[mult %d] col %d \n", __LINE__, col);
					if (rhs.num_nonzeros_by_row[col] != 0) 
					{
						complex_t accum = zero;
			  	  if(print_mult) ("[mult %d] row %u col %u accum: %.2e + %.2e i \n", __LINE__, row, col, get_real(accum), get_imag(accum));	
						for (uint32_t j = 0; j < this -> max_nnz_in_a_row; j++)
						{
							uint32_t col_loc = this->nonzero_col_locations[row][j];

  					  if(print_mult) ("[mult %d] row %u col_loc %u \n", __LINE__, row, col_loc);
							if(col_loc != num_cols)
							{
								for (uint32_t k=0; k < rhs.max_nnz_in_a_row; k++)
								{
									if (col_loc == rhs.nonzero_col_locations[col][k])
									{
										if(print_mult) ("[mult %d] Match k=%u \t col_loc=%u \n", __LINE__, k, col_loc);
                    complex_t this_val = this->nonzero_values[row][j];
                    complex_t rhs_val = (rhs.nonzero_values[col][k]) *conj;
										if(print_mult) ("[mult %d] \n\taccum = %.2e + %.2e i \n\tthis val  = %.2e + %.2e i \n\trhs val   = %.2e + %.2e i \n", __LINE__, get_real(accum), get_imag(accum), get_real(this_val), get_imag(this_val), get_real(rhs_val), get_imag(rhs_val));
										complex_t product = mul(this_val, rhs_val);
										if(print_mult) ("[mult %d] product = %.2e + %.2e i \n", __LINE__, get_real(product), get_imag(product));
										
										accum = add(accum, mul(this->nonzero_values[row][j], rhs.nonzero_values[col][k]*conj ));					
									  if(print_mult) ("[mult %d] row %u col_loc  %u \t Accum now %.2e + %.2e i \n", __LINE__,row, col_loc, get_real(accum), get_imag(accum));
									}					
								}
							}
						}

            if(get_real(accum)!=0.0 || get_imag(accum)!=0.0)
            {
              if(print_mult) ("[mult %d] Accum nonzero. Row %u Col %u Val %.2e + %.2e i \n", __LINE__, row, col, get_real(accum), get_imag(accum));
              nonzero_values[k] = accum;
              rows[k] = row;
              cols[k] = col;
              temp_nnz_by_row[row]++;

              k++;
              total_num_nnz ++;
if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

              if(row!=col)
              {
                if(print_mult) ("[mult %d] Accum nonzero; row!=col. Row %u Col %u Val %.2e + %.2e i \n", __LINE__, col, row, get_real(accum), get_imag(accum*conj));
                nonzero_values[k] = accum * conj;
                rows[k] = col;
                cols[k] = row;  
                k++;
                total_num_nnz ++;
                temp_nnz_by_row[col]++;
              }
if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);
              
            }
            
//						dst_row[col] = accum;
//						dst[col][row] = (accum*conj); 

					}
				}
			}
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);
			 
	  } // end for (row) loop

    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

    uint32_t max_nnz = 0;
    for(uint32_t a=0; a<num_rows; a++)
    {
      if(temp_nnz_by_row[a] > max_nnz)
      {
        max_nnz = temp_nnz_by_row[a];
      }
    }
    if(print_mult) ("[mult %d] Max NNZ = %u \n", __LINE__, max_nnz);
    for(uint32_t i=0; i<num_rows;i++)
    {
      row_full_upto[i] = 0;
    }
  
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);


    uint32_t* tmp_nnz_by_row;
    tmp_nnz_by_row = new uint32_t[num_rows];
    
    complex_t** tmp_nnz_vals;
    tmp_nnz_vals = new complex_t*[num_rows];
    uint32_t** tmp_nnz_col_locs;
    tmp_nnz_col_locs = new uint32_t*[num_rows];
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);
    
    for(uint32_t i=0; i<num_rows; i++)
    {
      tmp_nnz_vals[i] = new complex_t[max_nnz];
      tmp_nnz_col_locs[i] = new uint32_t[max_nnz];
    }
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);
    
    uint32_t a = 0;
    for(uint32_t k=0; k<total_num_nnz; k++)
    {
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);
    
      uint32_t row = rows[k];
      uint32_t col = cols[k];
      complex_t val = nonzero_values[k];
      a = row_full_upto[row];
      
      tmp_nnz_vals[row][a] = val;
      tmp_nnz_col_locs[row][a] = col;
      row_full_upto[row]++;
    }
    /*
    for(uint32_t i=0; i<num_rows; i++)
    {
      for(uint32_t j=tmp_nnz_by_row[i]; j< 
    }
    */
    uint32_t tmp_max_nnz = 0;
    for(uint32_t i=0; i<num_rows; i++)
    {
      tmp_nnz_by_row[i] = temp_nnz_by_row[i];
      if(tmp_nnz_by_row[i] > tmp_max_nnz)
      {
        tmp_max_nnz = tmp_nnz_by_row[i];
      }
    }

    if(print_mult) ("[mult %d] \n\t tmp_max_nnz = %u \n\t Tmp Values: \n", __LINE__, tmp_max_nnz);
    for(uint32_t i=0; i<num_rows; i++)
    {     
      if(print_mult) ("[mult %d] Row %u has %u nnz vals \n", __LINE__, i, tmp_nnz_by_row[i]);
      for(uint32_t j=0; j<tmp_max_nnz; j++)
      {
        complex_t val = tmp_nnz_vals[i][j];
        uint32_t col = tmp_nnz_col_locs[i][j];
        if(print_mult) ("[mult %d] Col loc: %u \t val: %.2e+%.2ei \n", __LINE__, col, get_real(val), get_imag(val));
      }
      
    }

    if(print_mult) ("[mult %d] New max_nnz going into reallocate fnc = %u \n", __LINE__, tmp_max_nnz);

    dst.reallocate(tmp_max_nnz, tmp_nnz_by_row, tmp_nnz_vals, tmp_nnz_col_locs);    

    if(PRINT_MATRICES_MULT)
    {
      if(print_mult) ("[mult %d] Result of MULT : \n", __LINE__);
      dst.print_compressed_storage_full();
    }

if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

}





void ComplexMatrix::mul_herm_for_e_minus_i(const ComplexMatrix& rhs, ComplexMatrix& dst) // changing const of rhs
{
    // TODO: take advantage of the fact that these are diagonally semmetrical
    size_t size = num_rows;
    complex_t zero = to_complex(0.0, 0.0);
    complex_t conj = to_complex(1.0, -1.0);

#if OPT_4
		dst.make_zero();
    for (uint32_t row = 0; row < size; ++row)
    {
			if(this->num_nonzeros_by_row[row] != 0)
			{
				complex_t* dst_row = dst.get_row(row);
				for (uint32_t col = row; col<size; ++col)
				{
					if (rhs.num_nonzeros_by_row[col] != 0) 
					{
						complex_t accum = zero;
			  		
						for (uint32_t j = 0; j < this -> max_nnz_in_a_row; j++)
						{
							uint32_t col_loc = this->nonzero_col_locations[row][j];
							if(col_loc != num_cols)
							{
								for (uint32_t k=0; k < rhs.max_nnz_in_a_row; k++)
								{
									if (col_loc == rhs.nonzero_col_locations[col][k])
									{
										accum = add(accum, mul(this->nonzero_values[row][j], rhs.nonzero_values[col][k]*conj ));					
									}					
								}
							}
						}
            // instead of filling here, fill nnz_array ??
						dst_row[col] = accum;
						dst[col][row] = (accum*conj); 

					}
					}
				} // end for (row) loop

		}

#elif OPT_3

		for (uint32_t row = 0; row < size; ++row)
		{
      // Here we make the assumption that the output will be symmetric across
      // the diagonal, so we only need to calculate half of it, and  then
      // write the other half to match.
      complex_t* dst_row = dst.get_row(row);
      const complex_t* src1 = get_row(row);
      for (uint32_t col = row; col < size; ++col)
      {
          const complex_t* src2 = rhs.get_row(col);
          complex_t accum = zero;
          for (uint32_t i = 0; i < size; ++i)
              accum = add(accum, mul(src1[i], src2[i] * conj));
          dst_row[col] = accum;
          dst[col][row] = accum * conj; // This * conj may belong on the previous line
      }
    } // end for (row) loop
#elif mul_full
		for (uint32_t row = 0; row < size; ++row)
		{
      complex_t* dst_row = dst.get_row(row);
			for (uint32_t col =0; col<size; ++col)
			{
				complex_t accum = zero;
				for (uint32_t i = 0; i < size; ++i)
				{
					accum = add(accum, mul((*this)[row][i], rhs[i][col]));
				}
				dst_row[col] = accum;
			}
		}
#endif // end if opt4, opt3 
}


// this += rhs * scale
void ComplexMatrix::add_scaled_hermitian(const ComplexMatrix& rhs, const scalar_t& scale)
{
    for (size_t i = 0; i < num_rows * num_rows; ++i)
        values[i] = add(values[i], mul_scalar(rhs.values[i], scale));
}

void ComplexMatrix::add_complex_scaled_hermitian(const ComplexMatrix& rhs, const complex_t& scale)
{
    for (size_t i = 0; i < num_rows * num_rows; ++i)
        values[i] = add(values[i], mul(rhs.values[i], scale));
}


void ComplexMatrix::add_complex_scaled_hermitian_sparse(const ComplexMatrix& rhs, const complex_t& scale)
{

    bool print_add = true;
    if(print_add) printf("[addition fnc] THIS: \n");
    if(print_add) this -> print_compressed_storage_full();
    if(print_add) printf("[addition fnc] RHS: \n");
    if(print_add) rhs.print_compressed_storage_full();
    /* Set up arrays/pointers etc. to send to reallocate function at end. */
    uint32_t num_rows = this->num_rows;
    printf("Num rows: %u \n", num_rows);
    uint32_t upper_max_nnz = num_rows;
    printf("upper max nnz = %u \n", upper_max_nnz);
    uint32_t new_max_nnz=0;
    uint32_t* tmp_nnz_by_row;
    tmp_nnz_by_row = new uint32_t[num_rows];
    
    complex_t** tmp_nnz_vals;
    tmp_nnz_vals = new complex_t*[num_rows];
    uint32_t** tmp_nnz_col_locs;
    tmp_nnz_col_locs = new uint32_t*[num_rows];
    
    for(uint32_t i=0; i<num_rows; i++)
    {
      tmp_nnz_vals[i] = new complex_t[upper_max_nnz];
      tmp_nnz_col_locs[i] = new uint32_t[upper_max_nnz];
      tmp_nnz_by_row[i] = 0;
    }

    /* Actual addition */
    
    for(uint32_t row=0; row<num_rows; row++)
    {
      uint32_t a;
      uint32_t b;
      uint32_t c;
      uint32_t rhs_unused_cols[num_rows];
      uint32_t this_unused_cols[num_rows];
      uint32_t this_row_num_nnz = this -> num_nonzeros_by_row[row];
      uint32_t rhs_row_num_nnz = rhs.num_nonzeros_by_row[row];
      uint32_t rhs_not_added_by_col[rhs_row_num_nnz];
      uint32_t this_not_added_by_col[this_row_num_nnz];

      uint32_t row_nnz_counter = 0;
      if(print_add) printf("[addition fnc] Row %u: this_nnz: %u \t rhs_nnz: %u \n", row, this_row_num_nnz, rhs_row_num_nnz);
      a=b=c=0;
      if(print_add) printf("[addition fnc] Before row loop: a=%u \t b=%u \t c=%u \n", a,b,c);
      uint32_t rhs_cols_been_added[rhs_row_num_nnz];
      for(uint32_t i=0; i<rhs_row_num_nnz; i++)
      {
        rhs_cols_been_added[i] = 0;
        rhs_not_added_by_col[i] = 0;
      }
      
      if (this_row_num_nnz == 0 )
      {
        printf("Row %u has 0 entries (%u) in LHS and %u in RHS \n", row, this->num_nonzeros_by_row[row], rhs_row_num_nnz);
        for(uint32_t j=0; j<rhs_row_num_nnz; j++)
        {
            rhs_unused_cols[b] = j;
            printf("ROW %u RHS col -- j= %u not matched. b++\n",row,j);
            b++;
        }
      }
      else
      {
        for(uint32_t i=0; i<this_row_num_nnz; i++)
        {
          uint32_t this_col = this->nonzero_col_locations[row][i];
          bool this_col_matched = 0;
          uint32_t rhs_col_unused_counter;
          uint32_t only_do_once;
          uint32_t this_col_unused_counter;
        
          for(uint32_t j=0; j<rhs_row_num_nnz; j++)
          {
            if(j==0) 
            {
              rhs_col_unused_counter = 0;
              only_do_once = 0;
            }

            uint32_t rhs_col = rhs.nonzero_col_locations[row][j];
            if(print_add) printf("[addition fnc] row %u \t this col = %u rhs col = %u \n", row, this_col, rhs_col);
            
            if(this_col == rhs_col)
            {
              tmp_nnz_vals[row][a] = this->nonzero_values[row][i] + mul(rhs.nonzero_values[row][j], scale);
              tmp_nnz_col_locs[row][a] = this_col;
              a++;
              printf("== a++ \n");
              this_col_matched = 1;
              printf("ROW %u COL %u matched a++ \n",row, this_col);
              row_nnz_counter++;
              tmp_nnz_by_row[row]++;
              rhs_cols_been_added[j] = 1;
            }
            else
            {
               rhs_not_added_by_col[j]++;
            }

            if(rhs_not_added_by_col[j] == this_row_num_nnz)
            {
              rhs_unused_cols[b] = j;
              rhs_cols_been_added[j] = 1;
              b++;
              printf("ROW %u  RHS col %u not matched. b++\n",row, rhs_col);
            }
            
          }

          if(this_col_matched == 0)
          {
              this_unused_cols[c] = i;
              printf("ROW %u THIS col %u not matched. b++\n",row, this_col);
              c++;
          }
        }
      }    

      if(print_add) printf("[addition fnc] Before unused arrays, a=%u b=%u c=%u \n", a,b,c);
      for(uint32_t k=0; k<b; k++)
      {
        uint32_t col = rhs_unused_cols[k];
        if(print_add) printf("[addition fnc] row %u unused RHS col %u \n", row, rhs.nonzero_col_locations[row][col]);
        tmp_nnz_vals[row][a] = mul(rhs.nonzero_values[row][col], scale);
        tmp_nnz_col_locs[row][a] = rhs.nonzero_col_locations[row][col];
        a++;
        printf("k<b a++ \n");
        row_nnz_counter++;
        tmp_nnz_by_row[row]++;

      } 

      for(uint32_t k=0; k<c; k++)
      {
        uint32_t col = this_unused_cols[k];
        if(print_add) ("[addition fnc] row %u unused this col %u \n", row, this->nonzero_col_locations[row][col]);
        tmp_nnz_vals[row][a] = this->nonzero_values[row][col];
        tmp_nnz_col_locs[row][a] = this->nonzero_col_locations[row][col];
        a++;
        printf("k<c a++ \n");
        row_nnz_counter++;
        tmp_nnz_by_row[row]++;
      } 
      if(print_add) ("[addition fnc] After Used/Unused: a=%u \t b=%u \t c=%u \n", a,b,c);
      if(print_add) ("[addition fnc] Row %u Counter %u \n", row, row_nnz_counter);
    }
    
    new_max_nnz = 0;
    for(uint32_t i=0; i<num_rows;i++)
    {
      printf("[tmp_nnz_by_row] i=%u \t nnz= %u \n", i, tmp_nnz_by_row[i]);
      if(tmp_nnz_by_row[i] > new_max_nnz)
      {
        new_max_nnz = tmp_nnz_by_row[i];
      }
    }
    
    if(print_add) printf("[addition fnc %d] New max_nnz going into reallocate fnc = %u \n", __LINE__, new_max_nnz);
    this->reallocate(new_max_nnz, tmp_nnz_by_row, tmp_nnz_vals, tmp_nnz_col_locs);
    if(print_add) printf("[addition fnc %d] After Reallocation: \n", __LINE__);
    if(print_add) this-> print_compressed_storage_full();

}


void ComplexMatrix::add_hermitian(const ComplexMatrix& rhs)
{
	for(size_t i = 0; i<num_rows*num_cols; i++)
		values[i] = add(values[i], rhs.values[i]);
}


void ComplexMatrix::swap_matrices(ComplexMatrix& other)
{
  uint32_t num_rows = this-> num_rows;


  uint32_t this_max_nnz = this-> max_nnz_in_a_row;
  uint32_t* this_nnz_by_row;
  this_nnz_by_row = new uint32_t[num_rows];
  
  complex_t** this_nnz_vals;
  this_nnz_vals = new complex_t*[num_rows];
  uint32_t** this_nnz_col_locs;
  this_nnz_col_locs = new uint32_t*[num_rows];
  
  for(uint32_t i=0; i<num_rows; i++)
  {
    this_nnz_vals[i] = new complex_t[this_max_nnz];
    this_nnz_col_locs[i] = new uint32_t[this_max_nnz];
  }
  
  for(uint32_t i=0; i<num_rows; i++)
  {
    this_nnz_by_row[i] = this->num_nonzeros_by_row[i];
    for(uint32_t j=0; j<this->num_nonzeros_by_row[i]; j++)
    {
      this_nnz_vals[i][j] = this->nonzero_values[i][j];
      this_nnz_col_locs[i][j] = this->nonzero_col_locations[i][j];
    }
  } 


  uint32_t other_max_nnz = this-> max_nnz_in_a_row;
  uint32_t* other_nnz_by_row;
  other_nnz_by_row = new uint32_t[num_rows];
  
  complex_t** other_nnz_vals;
  other_nnz_vals = new complex_t*[num_rows];
  uint32_t** other_nnz_col_locs;
  other_nnz_col_locs = new uint32_t*[num_rows];
  
  for(uint32_t i=0; i<num_rows; i++)
  {
    other_nnz_vals[i] = new complex_t[other_max_nnz];
    other_nnz_col_locs[i] = new uint32_t[other_max_nnz];
  }
  
  for(uint32_t i=0; i<num_rows; i++)
  {
    other_nnz_by_row[i] = other.num_nonzeros_by_row[i];
    for(uint32_t j=0; j< other.num_nonzeros_by_row[i]; j++)
    {
      other_nnz_vals[i][j] = other.nonzero_values[i][j];
      other_nnz_col_locs[i][j] = other.nonzero_col_locations[i][j];
    }
  } 

  this->reallocate(other_max_nnz, other_nnz_by_row, other_nnz_vals, other_nnz_col_locs);    
  other.reallocate(this_max_nnz, this_nnz_by_row, this_nnz_vals, this_nnz_col_locs);    
  
}



void ComplexMatrix::steal_values(const ComplexMatrix& other)
{
  uint32_t num_rows = this-> num_rows;

  uint32_t other_max_nnz = this-> max_nnz_in_a_row;
  uint32_t* other_nnz_by_row;
  other_nnz_by_row = new uint32_t[num_rows];
  
  complex_t** other_nnz_vals;
  other_nnz_vals = new complex_t*[num_rows];
  uint32_t** other_nnz_col_locs;
  other_nnz_col_locs = new uint32_t*[num_rows];
  
  for(uint32_t i=0; i<num_rows; i++)
  {
    other_nnz_vals[i] = new complex_t[other_max_nnz];
    other_nnz_col_locs[i] = new uint32_t[other_max_nnz];
  }
  
  for(uint32_t i=0; i<num_rows; i++)
  {
    other_nnz_by_row[i] = other.num_nonzeros_by_row[i];
    for(uint32_t j=0; j< other.num_nonzeros_by_row[i]; j++)
    {
      other_nnz_vals[i][j] = other.nonzero_values[i][j];
      other_nnz_col_locs[i][j] = other.nonzero_col_locations[i][j];
    }
  } 

  this->reallocate(other_max_nnz, other_nnz_by_row, other_nnz_vals, other_nnz_col_locs);    
  
}



bool ComplexMatrix::exp_ham_sparse(ComplexMatrix& dst, double scale, double precision, bool plus_minus) const
{
    /* To avoid extra copying, we alternate power accumulation matrices */
    if(PRINT_LINE_DEBUG) printf("In exp ham: \n");
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

    double scalar_by_time = scale;
		bool infinite_val = false; // If the matrix multiplication doesn't diverge, this is set to true and returned to indicate the method has failed. 
    bool rescale_method = true; // Flag to rescale Hamiltonian so that all elements <=1
    double norm_scalar;
    bool do_print = false;
    bool print_exp_k_loop = false;
bool print_exp_mult = true;
    uint32_t k_max = 34;
    printf("Num rows: %u cols: %u \n", num_rows, num_cols);
    /* TODO Construct pa[0,1] as sparse from the start */
    ComplexMatrix power_accumulator0(num_rows, num_cols);
    ComplexMatrix power_accumulator1(num_rows, num_cols);
    power_accumulator0.make_identity();
    power_accumulator1.make_identity();
    ComplexMatrix* pa[2] = {&power_accumulator0, &power_accumulator1};



    dst.make_zero();
    dst.compress_matrix_storage();
    printf("At beginning: dst \n");
    dst.print_compressed_storage_full();

    printf("pa[0]: \n");
    pa[0]->debug_print();
    printf("pa[1]: \n");
    pa[1]->debug_print();

    double k_fact = 1.0;
		double scale_time_over_k_factorial = 1.0;
		// double current_max_element = this -> get_max_element_magnitude();
    bool done = false;
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

     ComplexMatrix& new_pa = *pa[0];
     ComplexMatrix& old_pa = *pa[1];

    new_pa.compress_matrix_storage();
    old_pa.compress_matrix_storage();
    for (uint32_t k = 0; !done; ++k)
    {
        if (k > 0)
      	{
						k_fact /= k;
						scale_time_over_k_factorial *= scalar_by_time/k;
				}
				
//        if (scale_time_over_k_factorial * current_max_element >= precision)
        if (scale_time_over_k_factorial >= precision)
        { 
        /* 
        * This is where actual exponentiation happens by multiplying a running total,
        * H^m, by H to get H^m. 
        * This is then multiplied by (t*s)^m/m! and added to become the new running total.
        * Here s is a scalar - the largest magnitude of any matrix element. This is factored out 
        * of the matrix so that all values inside the matrix are less than one, 
        * to keep the multiplication from diverging and introducing matrix elements 
        * larger than the computer can handle.
        * t is the time set by the heuristic. 
        *
        */
        
            uint32_t alternate = k & 1;

           /* TODO [swap] remove declaration to just below pa declaraion */
//           ComplexMatrix& new_pa = *pa[alternate];
 //          ComplexMatrix& old_pa = *pa[1 - alternate];
/* 
            printf("[exp k loop k= %u %d] new_pa: \n", k, __LINE__);
            new_pa.debug_print();

            printf("[exp k loop k= %u %d] new_pa: \n", k, __LINE__);
            new_pa.debug_print();


            new_pa.compress_matrix_storage();
            old_pa.compress_matrix_storage();
            printf("[k=%u] Compressed before exp \n", k);
            printf("new_pa:\n");
            new_pa.print_compressed_storage();
            printf("old_pa:\n");
            old_pa.print_compressed_storage();
*/

            if (k > 0)
            {
                //new_pa.compress_matrix_storage();
                //old_pa.compress_matrix_storage();
								//new_pa.make_zero();
                if(k>1)
                {
                  if(print_exp_k_loop) printf("[exp k=%u %d] BEFORE SWAP \n", k, __LINE__);
                  if(print_exp_k_loop) printf("old_pa: \n");
                  if(print_exp_k_loop) old_pa.print_compressed_storage_full();
                  if(print_exp_k_loop) printf("new_pa: \n");
                  if(print_exp_k_loop) new_pa.print_compressed_storage_full();

                  old_pa.swap_matrices(new_pa);

                  if(print_exp_k_loop) printf("[exp k=%u %d] AFTER SWAP \n", k, __LINE__);
                  if(print_exp_k_loop) printf("old_pa: \n");
                  if(print_exp_k_loop) old_pa.print_compressed_storage_full();
                  if(print_exp_k_loop) printf("new_pa: \n");
                  if(print_exp_k_loop) new_pa.print_compressed_storage_full();

                }

								if(print_exp_k_loop | print_exp_mult) printf("[exp mult k=%u %d] Multiplying:  \n", k, __LINE__);
								if(print_exp_k_loop | print_exp_mult) printf("[exp mult k=%u %d] old_pa: \n", k, __LINE__);
								if(print_exp_k_loop | print_exp_mult) old_pa.print_compressed_storage_full();
								if(print_exp_k_loop | print_exp_mult) printf("[exp mult k=%u %d] by THIS: \n", k, __LINE__);
								this->print_compressed_storage_full();



                
                old_pa.sparse_hermitian_mult(*this, new_pa);                



								if(print_exp_k_loop | print_exp_mult) printf("[exp mult k=%u %d] After mult: new_pa: \n",k,  __LINE__);
								if(print_exp_k_loop | print_exp_mult) new_pa.print_compressed_storage_full();	
  
                /* TODO [swap] 
                old_pa.swap(new_pa) 
                */
                

						}	
						
            complex_t one_over_k_factorial_simd;
            
            /* Set symmetrical element */
            //printf("plus_minus = %u\n", plus_minus);
            if(plus_minus == true) // plus_minus = true -> (+i) 
            {
						  if((k)%4 == 0 ) // k=0,4,8...
						  {
							  one_over_k_factorial_simd = to_complex(scale_time_over_k_factorial, 0.0); 
						  }
						  else if ( (k+3)%4 ==0) // k = 1,5,9
						  {
							  one_over_k_factorial_simd = to_complex(0.0, 1.0*scale_time_over_k_factorial); 
						  }
						  else if ((k+2)%4 == 0) // k = 2,6,10
						  {
							  one_over_k_factorial_simd = to_complex(-1.0*scale_time_over_k_factorial, 0.0); 
						  }
						  else if ((k+1)%4 == 0 ) // k =3, 7, 11
						  {
							  one_over_k_factorial_simd = to_complex(0.0, -1.0*scale_time_over_k_factorial); 
						  }					
            }
            else
            { // plus_minus = false -> (-i)
						  if((k)%4 == 0 ) // k=0,4,8...
						  {
							  one_over_k_factorial_simd = to_complex(scale_time_over_k_factorial, 0.0); 
						  }
						  else if ( (k+3)%4 ==0) // k = 1,5,9
						  {
							  one_over_k_factorial_simd = to_complex(0.0, -1.0*scale_time_over_k_factorial); 
						  }
						  else if ((k+2)%4 == 0) // k = 2,6,10
						  {
							  one_over_k_factorial_simd = to_complex(-1.0*scale_time_over_k_factorial, 0.0); 
						  }
						  else if ((k+1)%4 == 0 ) // k =3, 7, 11
						  {
							  one_over_k_factorial_simd = to_complex(0.0, 1.0*scale_time_over_k_factorial); 
						  }					
						  else 
						  {
							  printf("k = %u doesn't meet criteria.\n", k);
						  }
            }
    
            if(!std::isfinite(scale_time_over_k_factorial) || k_fact < 1e-300)
            {
            /* 
            * If values are intractable using double floating point precision,
            * fail the process and the function returns 1 to indicate failure.
            */
            	done = true;
            	infinite_val = true;
            }
            //else if (scale_time_over_k_factorial < precision)

            else if (scale_time_over_k_factorial < precision || k>k_max)
            {
            	done = true;
            }
            
            else
            { /* only add to destination matrix if not yet at inf */
              printf("[exp addition k= %u]  adding new_pa by scalar %.2e + %.2e: \n",k, get_real(one_over_k_factorial_simd), get_imag(one_over_k_factorial_simd));
              new_pa.print_compressed_storage_full();
              printf("[exp addition k= %u] adding to dst: \n",k);
              dst.print_compressed_storage_full();
		          dst.add_complex_scaled_hermitian_sparse(new_pa, one_over_k_factorial_simd);
              printf("[exp addition k= %u] After addition, dst: \n",k);
	            printf("[sparse exp %d] adding. k=%u \n", __LINE__, k);
              dst.print_compressed_storage_full();
            }
            
        }
        else
        {
            done = true;
        }

    printf("[exp k=%u line %d] End of step. DST is now: \n", k, __LINE__);
    dst.print_compressed_storage_full();

    }
    //dst.compress_matrix_storage();
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);
    printf("\n \n \nEND OF EXP: RESULT \n");
    dst.print_compressed_storage_full();
    return infinite_val;
}

// TODO separate fnc: sparse_exp_ham
bool ComplexMatrix::exp_ham(ComplexMatrix& dst, double scale, double precision, bool plus_minus) const
{
    /* To avoid extra copying, we alternate power accumulation matrices */
    if(PRINT_LINE_DEBUG) printf("In exp ham: \n");
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

    double scalar_by_time = scale;
		bool infinite_val = false; // If the matrix multiplication doesn't diverge, this is set to true and returned to indicate the method has failed. 
    double norm_scalar;
    bool do_print = false;

    printf("Num rows: %u cols: %u \n", num_rows, num_cols);
    ComplexMatrix power_accumulator0(num_rows, num_cols);
    ComplexMatrix power_accumulator1(num_rows, num_cols);
    power_accumulator0.make_identity();
    power_accumulator1.make_identity();
    ComplexMatrix* pa[2] = {&power_accumulator0, &power_accumulator1};

    dst.make_zero();

    double k_fact = 1.0;
		double scale_time_over_k_factorial = 1.0;
		// double current_max_element = this -> get_max_element_magnitude();
    bool done = false;
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

    for (uint32_t k = 0; !done; ++k)
    {
        if (k > 0)
      	{
						k_fact /= k;
						scale_time_over_k_factorial *= scalar_by_time/k;
				}
				
        if (scale_time_over_k_factorial >= precision)
        { 
        /* 
        * This is where actual exponentiation happens by multiplying a running total,
        * H^m, by H to get H^m. 
        * This is then multiplied by (t*s)^m/m! and added to become the new running total.
        * Here s is a scalar - the largest magnitude of any matrix element. This is factored out 
        * of the matrix so that all values inside the matrix are less than one, 
        * to keep the multiplication from diverging and introducing matrix elements 
        * larger than the computer can handle.
        * t is the time set by the heuristic. 
        *
        */
        
            uint32_t alternate = k & 1;
            ComplexMatrix& new_pa = *pa[alternate];
            ComplexMatrix& old_pa = *pa[1 - alternate];


            if (k > 0)
            {
                new_pa.compress_matrix_storage();
                old_pa.compress_matrix_storage();
									
                old_pa.mul_herm_for_e_minus_i(*this, new_pa);                
						}	
						
            complex_t one_over_k_factorial_simd;
            
            /* Set symmetrical element */
            //printf("plus_minus = %u\n", plus_minus);
            if(plus_minus == true) // plus_minus = true -> (+i) 
            {
						  if((k)%4 == 0 ) // k=0,4,8...
						  {
							  one_over_k_factorial_simd = to_complex(scale_time_over_k_factorial, 0.0); 
						  }
						  else if ( (k+3)%4 ==0) // k = 1,5,9
						  {
							  one_over_k_factorial_simd = to_complex(0.0, 1.0*scale_time_over_k_factorial); 
						  }
						  else if ((k+2)%4 == 0) // k = 2,6,10
						  {
							  one_over_k_factorial_simd = to_complex(-1.0*scale_time_over_k_factorial, 0.0); 
						  }
						  else if ((k+1)%4 == 0 ) // k =3, 7, 11
						  {
							  one_over_k_factorial_simd = to_complex(0.0, -1.0*scale_time_over_k_factorial); 
						  }					
            }
            else
            { // plus_minus = false -> (-i)
						  if((k)%4 == 0 ) // k=0,4,8...
						  {
							  one_over_k_factorial_simd = to_complex(scale_time_over_k_factorial, 0.0); 
						  }
						  else if ( (k+3)%4 ==0) // k = 1,5,9
						  {
							  one_over_k_factorial_simd = to_complex(0.0, -1.0*scale_time_over_k_factorial); 
						  }
						  else if ((k+2)%4 == 0) // k = 2,6,10
						  {
							  one_over_k_factorial_simd = to_complex(-1.0*scale_time_over_k_factorial, 0.0); 
						  }
						  else if ((k+1)%4 == 0 ) // k =3, 7, 11
						  {
							  one_over_k_factorial_simd = to_complex(0.0, 1.0*scale_time_over_k_factorial); 
						  }					
						  else 
						  {
							  printf("k = %u doesn't meet criteria.\n", k);
						  }
            }
    
            if(!std::isfinite(scale_time_over_k_factorial) || k_fact < 1e-300)
            {
            /* 
            * If values are intractable using double floating point precision,
            * fail the process and the function returns 1 to indicate failure.
            */
            	done = true;
            	infinite_val = true;
            }
            else if (scale_time_over_k_factorial < precision)
            {
            	done = true;
            }
            
            else
            { /* only add to destination matrix if not yet at inf */
		          dst.add_complex_scaled_hermitian(new_pa, one_over_k_factorial_simd);
            }
            
        }
        else
        {
            done = true;
        }
    }
    dst.compress_matrix_storage();
    if(PRINT_LINE_DEBUG) printf("Line %d in file %s \n", __LINE__, __FILE__);

    return infinite_val;
}


void ComplexMatrix::debug_print() const
{
//		printf("---- -- Debug Print -- ----\n");

    for (uint32_t row = 0; row < num_rows; ++row)
    {
        const complex_t* prow = get_row(row);
        printf("  | ");
        for (uint32_t col = 0; col < num_cols; ++col)
        {
            complex_t val = prow[col];
            double re = get_real(val);
            double im = get_imag(val);
            if (re || im)
                printf("%.7e+%.7ei ", get_real(val), get_imag(val));
            else
                printf("     0     ");
        }
        printf(" |\n");
    }
	printf("---- ---- ----\n");
}
