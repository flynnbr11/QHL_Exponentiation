// Copyright 2017 University of Bristol
#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <random>
#include <vector>
typedef __m128d scalar_t;
typedef __m128d complex_t;
#define TOLERANCE 1e-15

#define VERBOSE_H 0
#define COMPRESS 1

inline complex_t load_complex(const complex_t* addr) {
    return _mm_load_pd((const double*)addr);
}
inline scalar_t to_scalar(double rhs) { return _mm_set_pd(rhs, rhs); }
inline complex_t to_complex(double r, double i) { return _mm_set_pd(i, r); }
inline double get_real(const complex_t c) { return ((double*)&c)[0]; }
inline double get_imag(const complex_t c) { return ((double*)&c)[1]; }
inline double get_scalar_val_0(const scalar_t c) {return ((double*)&c)[0];}
inline double get_scalar_val_1(const scalar_t c) {return ((double*)&c)[1];}

inline complex_t complex_conjugate(const complex_t& c) { return to_complex(get_real(c), -get_imag(c)); }
inline const complex_t add(const complex_t lhs, const complex_t rhs) { return _mm_add_pd(lhs, rhs); }
inline const complex_t sub(const complex_t lhs, const complex_t rhs) { return _mm_sub_pd(lhs, rhs); }
inline const complex_t mul_scalar(const complex_t lhs, const scalar_t rhs) { return _mm_mul_pd(lhs, rhs); }
inline double mag_sqr(const complex_t lhs) { complex_t c = _mm_mul_pd(lhs, lhs); return get_real(c) + get_imag(c); }
inline void addeq(complex_t* lhs, const complex_t rhs) { *lhs = _mm_add_pd(*lhs, rhs); }
inline void subeq(complex_t* lhs, const complex_t rhs) { *lhs = _mm_sub_pd(*lhs, rhs); }
inline void muleq_scalar(complex_t* lhs, const scalar_t rhs) { *lhs = _mm_mul_pd(*lhs, rhs); }
inline const complex_t mul(const complex_t a_b, const complex_t c_d)
{
    complex_t a_a = _mm_permute_pd(a_b, 0);
    complex_t b_b = _mm_permute_pd(a_b, 3);
    complex_t d_c = _mm_permute_pd(c_d, 1);
    complex_t bc_bd = _mm_mul_pd(b_b, d_c);
    complex_t ad_ac = _mm_mul_pd(a_a, c_d);
    const complex_t outcome = _mm_addsub_pd(ad_ac, bc_bd);
		return outcome;
    //return _mm_addsub_pd(ad_ac, bc_bd);
}
inline void muleq(complex_t* a_b, const complex_t c_d)
{
    *a_b = mul(*a_b, c_d);
}

complex_t do_permanent(const complex_t* mtx_data, uint32_t size);
double do_permanent(const double* mtx_data, uint32_t size);

inline uint32_t int_log2(uint64_t val)
{
    uint32_t result = 0;
    val >>= 1;
    while (val)
    {
        val >>= 1;
        result++;
    }
    return result;
}
inline uint32_t bit_get(uint64_t val, uint64_t place)
{
    return (val >> place) & 1;
}

uint32_t random_int_inclusive(uint32_t low, uint32_t high);
double random_plus_minus();
double random_zero_to_one();

class RealMatrix;

class ComplexMatrix {
public:
    ComplexMatrix()
    : allocated_nnz_array(false), self_allocated(false), num_rows(0), num_cols(0), num_values(0), values(NULL)
    {
				if(VERBOSE_H) printf("In ComplexMatrix  default constructor  w/ no arguments \n");
    }
    ComplexMatrix(uint32_t rows, uint32_t cols)
    : allocated_nnz_array(false)
    {
				if(VERBOSE_H) printf("In ComplexMatrix constructor  w/ row and col \n");
        allocate(rows, cols);
    }
    ComplexMatrix(uint32_t rows, uint32_t cols, complex_t* data)
    : allocated_nnz_array(false), self_allocated(false), num_rows(rows), num_cols(cols), num_values(rows * cols), values(data)
    {
				if(VERBOSE_H) printf("In ComplexMatrix constructor  w/ vals \n");
    		// if(VERBOSE_H) debug_print();     		
        if(COMPRESS) compress_matrix_storage();
    }
    ~ComplexMatrix()
    {
        destroy();
    }
    complex_t permanent() const
    {
        return do_permanent(values, num_rows);
    }
    complex_t* get_row(uint32_t row)
    {
        return values + row * num_cols;
    }
		

    const complex_t* get_row(uint32_t row) const
    {
        return values + row * num_cols;
    }
    complex_t* operator[](uint32_t row)
    {
        return get_row(row);
    }
    const complex_t* operator[](uint32_t row) const
    {
        return get_row(row);
    }

		//*
		uint32_t sum_row(uint32_t row) const
		{
			const complex_t* row_vals = get_row(row);
			uint32_t running_sum = 0; 
			for(uint32_t i=0; i<num_cols; i++)
			{
				if(::mag_sqr(row_vals[i]) > TOLERANCE) // TODO: replace with comparison to complex zero
				{
					running_sum += 1;
				}
			}
			return running_sum;
		}
		//*/


		void compress_matrix_storage() 
		{
        complex_t complex_zero = to_complex(0.0, 0.0);
				
				if(allocated_nnz_array==1){
					delete[] num_nonzeros_by_row; 

					for (uint32_t i=0; i < num_rows; i++)
					{
						delete[] nonzero_col_locations[i]; 
						delete[] nonzero_values[i];
					}
					delete[] nonzero_col_locations;
					delete[] nonzero_values; 
				}
				allocated_nnz_array = 1;
				num_nonzeros_by_row = new uint32_t[num_rows];
    			max_nnz_in_a_row = 0;

				//if(VERBOSE_H) debug_print();
				
				for(uint32_t i=0; i<num_rows; i++) // set num_nonzeros_by_row array.
				{
					uint32_t this_row_sum = sum_row(i);
					num_nonzeros_by_row[i]= this_row_sum;
					if (this_row_sum > max_nnz_in_a_row)
					{
						max_nnz_in_a_row = this_row_sum;
					}
				}
				
				// after max_nnz_in_a_row is known, can fill other arrays. 

				nonzero_col_locations = new uint32_t*[num_rows];
				nonzero_values = new complex_t*[num_rows];

				for(uint32_t i=0; i<num_rows; i++)
				{
					nonzero_col_locations[i] = new uint32_t[max_nnz_in_a_row];
					nonzero_values[i] = new complex_t[max_nnz_in_a_row];
			
					const complex_t *this_row = get_row(i);
					
					uint32_t k = 0; 
					for(uint32_t j=0; j < num_cols; j++)
					{
						if(::mag_sqr(this_row[j]) > TOLERANCE) // if element nonzero, add to nonzero_values.  
						{
							nonzero_col_locations[i][k] = j;
							nonzero_values[i][k] = this_row[j];
							k++;
						}
					}
					for(uint32_t l=k; l < max_nnz_in_a_row; l++)
					{
						nonzero_col_locations[i][l] = num_cols; // indicate there is no more non zero elements
						nonzero_values[i][l] = complex_zero;
					} 		
				}
				
		}
		
    void allocate(uint32_t rows, uint32_t cols)
    {
        self_allocated = true;
        num_rows = rows;
        num_cols = cols;
        num_values = num_rows * num_cols;
#ifdef MSVC
        values = (complex_t*)_aligned_malloc(num_values * sizeof(complex_t), 16);
#else
        values = (complex_t*)memalign(16, num_values * sizeof(complex_t));
#endif
				if(COMPRESS) compress_matrix_storage();
    }
    void mag_sqr(RealMatrix& dst) const;
    void make_identity();
    void make_zero();
    void mul_hermitian(const ComplexMatrix& rhs, ComplexMatrix& dst);
    void add_scaled_hermitian(const ComplexMatrix& rhs, const complex_t& scale);
    void add_hermitian(const ComplexMatrix& rhs);
    void expm_special(ComplexMatrix& dst, double precision) const;
    void cos_plus_i_sin(ComplexMatrix& dst, double precision) const;
    void debug_print() const;
    void print_compressed_storage() const;

    uint32_t max_nnz_in_a_row;
		uint32_t *num_nonzeros_by_row;
		uint32_t **nonzero_col_locations;
		complex_t **nonzero_values;


private:
    void destroy()
    {
        if (self_allocated)
        {
#ifdef MSVC
            if (values)
                _aligned_free(values);
#else
            if (values)
                free(values);
#endif
        }
				if(allocated_nnz_array==1){
					delete[] num_nonzeros_by_row; 
					delete[] nonzero_col_locations;
					delete[] nonzero_values; 
				}
 				allocated_nnz_array= false;
        self_allocated = false;
        values = NULL;
    }
    bool allocated_nnz_array;
    bool self_allocated;    // We're responsible for freeing the values
    uint32_t num_rows;      // number of rows
    uint32_t num_cols;      // number of columns
    uint32_t num_values;    // num_rows * num_cols 
    complex_t* values;      // the actual storage
};

class RealMatrix {
public:
    RealMatrix()
    : num_rows(0), num_cols(0)
    {
    }
    RealMatrix(uint32_t rows, uint32_t cols)
    {
        allocate(rows, cols);
    }
    ~RealMatrix()
    {
    }
    double permanent() const
    {
        return do_permanent(&values[0], num_rows);
    }
    double* get_row(uint32_t row)
    {
        return &values[0] + row * num_cols;
    }
    const double* get_row(uint32_t row) const
    {
        return &values[0] + row * num_cols;
    }
    void allocate(uint32_t rows, uint32_t cols)
    {
        num_rows = rows;
        num_cols = cols;
        values.resize(num_rows * num_cols);
    }
private:
    std::vector<double> values;
    uint32_t num_rows;
    uint32_t num_cols;
};

#endif // MATRIX_UTILS_H

