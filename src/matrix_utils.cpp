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

#define VERBOSE 1

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
    for (uint32_t row = 0; row < num_rows; ++row)
    {
        complex_t* prow = get_row(row);
        for (uint32_t col = row; col < num_cols; ++col)
        {
            complex_t* pcol = get_row(col);
            prow[col] = pcol[row] = zero;
        }
    }
}

#define OPT_3 0 // opt 3 correctly exploits Symmetrical shape, but not sparsity. 
#define OPT_4 0 // opt 4 used for development of sparsity utility
#define testing_class 1
// dst = this * rhs
void ComplexMatrix::mul_hermitian(const ComplexMatrix& rhs, ComplexMatrix& dst) const
{
    // TODO: take advantage of the fact that these are diagonally semmetrical
    size_t size = num_rows;
    complex_t zero = to_complex(0.0, 0.0);
    complex_t conj = to_complex(1.0, -1.0);

#if testing_class 
		//this -> compress_matrix_storage() const;
		if (VERBOSE) printf("This Max num nnz : %u \n", this->max_nnz_in_a_row);
		if (VERBOSE) printf("%u\n", this->max_nnz_in_a_row);
    for (uint32_t row = 0; row < size; ++row)
    {
			//printf("Self num nnz : %u \n", this->max_nnz_in_a_row);
			uint32_t row_sum = this->num_nonzeros_by_row[row]; 
			uint32_t row_sum_fnc = this->sum_row(row); 
			if (VERBOSE) printf("Row = %u: \t NNz = %u \t Fnc: %u \n", row, row_sum, row_sum_fnc);
		}
		if (VERBOSE)
		{
			printf("LHS : \n");
			this -> debug_print();
			printf("RHS : \n");
			rhs.debug_print();
		}
#else 		
    for (uint32_t row = 0; row < size; ++row)
    {

#if OPT_4
			if (VERBOSE) printf("Self num nnz : %u \n", this->max_nnz_in_a_row);
			uint32_t row_sum = this->num_nonzeros_by_row[row]; 
			if (VERBOSE) printf("Row = % zu: NNz = %u \n", row, row_sum);
			
//			printf("This num nnz : %u \n", rhs.max_nnz_in_a_row); 

			//printf("opt In 4 development section\n");
			complex_t* dst_row = dst.get_row(row);
			const complex_t* src1 = get_row(row);


			// This should be applied to the self matrix which the multiply method is called on, NOT rhs... 
			//*
			if(rhs.num_nonzeros_by_row[row] != 0)
			{
				// get cols of nnz from idxs
		      for (uint32_t i = 0; i < size; ++i)
					{
						complex_t accum = zero;
						for(uint32_t j = 0; j < rhs.max_nnz_in_a_row; j++)
						{
							if(rhs.nonzero_col_locations[row][j]!=0)
							{ 
								uint32_t column_location = rhs.nonzero_col_locations[row][j];
								// printf("Col loc = %u \n", rhs.nonzero_col_locations[row][j]);
								if (column_location > row) //only want to do RHS of matrix
								{
			            const complex_t* src2 = rhs.get_row(column_location);
					        accum = add(accum, mul(src1[i], src2[i]));
								}
							}
				      dst_row[i] = accum;
				      dst[i][row] = accum * conj; // This * conj may belong on the previous line
						}
				}
			}
			else //if entire row was zero, set to zero immediately.
			{
	      for (uint32_t i = 0; i < size; ++i)
				{
		      dst_row[i] = zero;
		      dst[i][row] = zero; // This * conj may belong on the previous line
				}				
			}
			//*/
#elif OPT_3
				printf("In opt 3 development section\n");
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
#endif // end if opt4, opt3 
    }
#endif // end if testing_class
}

// this += rhs * scale
void ComplexMatrix::add_scaled_hermitian(const ComplexMatrix& rhs, const scalar_t& scale)
{
    // TODO: take advantage of the fact that these are diagonally semmetrical
    for (size_t i = 0; i < num_rows * num_rows; ++i)
        values[i] = add(values[i], mul_scalar(rhs.values[i], scale));
}

void ComplexMatrix::expm_special(ComplexMatrix& dst, double precision) const
{
    // To avoid extra copying, we alternate power accumulation matrices
    ComplexMatrix power_accumulator0(num_rows, num_cols);
    ComplexMatrix power_accumulator1(num_rows, num_cols);
    power_accumulator0.make_identity();
    power_accumulator1.make_identity();
    ComplexMatrix* pa[2] = {&power_accumulator0, &power_accumulator1};

    dst.make_zero();

    double one_over_k_factorial = 1.0;
    bool done = false;
    for (uint32_t k = 0; !done; ++k)
    {
        if (k > 1)
            one_over_k_factorial /= k;
        if (one_over_k_factorial >= precision)
        {
            uint32_t alternate = k & 1;
            ComplexMatrix& new_pa = *pa[alternate];
//            new_pa.compress_matrix_storage(); // Want to compress 
            const ComplexMatrix& old_pa = *pa[1 - alternate];
            if (k > 0)
            {
            		if(VERBOSE){
		          		printf("In expm special; k= %u \n", k);
		          		printf("Old_pa : \n");
		          		old_pa.debug_print();
		          		printf("This Before mult : \n");
	 	           		this -> debug_print();
								}
                
                old_pa.mul_hermitian(*this, new_pa);

								if(VERBOSE){
		          		printf("new_pa After mult : \n");
		          		new_pa.debug_print();
	          		}
						}	
            scalar_t one_over_k_factorial_simd = to_scalar(one_over_k_factorial);
            dst.add_scaled_hermitian(new_pa, one_over_k_factorial_simd);
        }
        else
        {
        		
            done = true;
        }
    }
}





/* Adapted from Brian Butler's Matlab implementation here:
 *  https://www.mathworks.com/matlabcentral/fileexchange/
 *   53784-matrix-permanent-using-nijenhuis-wilf-in-cmex/content/perman_mat.m
 */

#define MAX_PERM_SIZE 128

complex_t do_permanent(const complex_t* mtx_data, uint32_t size)
{
    if (size > MAX_PERM_SIZE)
    {
        printf("Error: MAX_PERM_SIZE needs to be increased.\n");
        return to_complex(0, 0);
    }
    complex_t xrow_storage[MAX_PERM_SIZE];

    complex_t* xrow = xrow_storage;
    scalar_t half = to_scalar(0.5);
    complex_t p = to_complex(1.0, 0.0);

    for (size_t i = 0; i < size; ++i)
    {
        complex_t sum = to_complex(0.0, 0.0);
        for (size_t j = 0; j < size; ++j)
        {
            addeq(&sum, load_complex(&mtx_data[j * size + i]));
        }
        xrow[i] = sub(load_complex(&mtx_data[(size - 1) * size + i]), mul_scalar(sum, half));
        muleq(&p, xrow[i]);
    }

    uint64_t tn11 = (1L << (size - 1)) - 1;
    uint64_t y_prev = 0;
    for (uint64_t i = 0; i < tn11; ++i)
    {
        uint64_t yi = (i+1) ^ ((i+1) >> 1);
        uint32_t zi = int_log2(yi ^ y_prev);
        scalar_t si = to_scalar(-1.0 + 2.0 * bit_get(yi, zi));

        y_prev = yi;

        complex_t prodx = to_complex(1.0, 0.0);
        uint32_t offset = zi * size;
        for (uint32_t j = 0; j < size; ++j)
        {
            complex_t rr = add(xrow[j], mul_scalar(load_complex(&mtx_data[offset + j]), si));
            xrow[j] = rr;
            muleq(&prodx, rr);
        }
        if (i & 1)
            addeq(&p, prodx);
        else
            subeq(&p, prodx);
    }

    addeq(&p, p);
    if (!(size & 1))
        muleq_scalar(&p, to_scalar(-1.0));
    return p;
}

double do_permanent(const double* mtx_data, uint32_t size)
{
    if (size > MAX_PERM_SIZE)
    {
        printf("Error: MAX_PERM_SIZE needs to be increased.\n");
        return 0;
    }
    double xrow_storage[MAX_PERM_SIZE];

    double* xrow = xrow_storage;
    double p = 1.0;

    for (size_t i = 0; i < size; ++i)
    {
        double sum = 0.0;
        for (size_t j = 0; j < size; ++j)
            sum += mtx_data[j * size + i];
        xrow[i] = mtx_data[(size - 1) * size + i] - sum * 0.5;
        p *= xrow[i];
    }

    uint64_t tn11 = (1L << (size - 1)) - 1;
    uint64_t y_prev = 0;
    for (uint64_t i = 0; i < tn11; ++i)
    {
        uint64_t yi = (i+1) ^ ((i+1) >> 1);
        uint32_t zi = int_log2(yi ^ y_prev);
        double si = -1.0 + 2.0 * bit_get(yi, zi);

        y_prev = yi;

        double prodx = 1.0;
        uint32_t offset = zi * size;
        for (uint32_t j = 0; j < size; ++j)
        {
            double rr = xrow[j] + mtx_data[offset + j] * si;
            xrow[j] = rr;
            prodx *= rr;
        }
        if (i & 1)
            p += prodx;
        else
            p -= prodx;
    }

    p += p;
    if (!(size & 1))
        p = -p;
    return p;
}

void ComplexMatrix::debug_print() const
{
		printf("---- -- Debug Print-- ----\n");

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
                printf("%.2f+%.2fi ", get_real(val), get_imag(val));
            else
                printf("     0     ");
        }
        printf(" |\n");
    }
	printf("---- ---- ----\n");
}
