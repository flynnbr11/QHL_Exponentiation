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
typedef __m128d Scalar;
typedef __m128d complex_t;

inline complex_t load_complex(const complex_t* addr) {
    return _mm_load_pd((const double*)addr);
}
inline Scalar to_scalar(double rhs) { return _mm_set_pd(rhs, rhs); }
inline complex_t to_complex(double r, double i) { return _mm_set_pd(i, r); }
inline double get_real(const complex_t c) { return ((double*)&c)[0]; }
inline double get_imag(const complex_t c) { return ((double*)&c)[1]; }
inline const complex_t add(const complex_t lhs, const complex_t rhs) { return _mm_add_pd(lhs, rhs); }
inline const complex_t sub(const complex_t lhs, const complex_t rhs) { return _mm_sub_pd(lhs, rhs); }
inline const complex_t mul_scalar(const complex_t lhs, const Scalar rhs) { return _mm_mul_pd(lhs, rhs); }
inline double mag_sqr(const complex_t lhs) { complex_t c = _mm_mul_pd(lhs, lhs); return get_real(c) + get_imag(c); }
inline void addeq(complex_t* lhs, const complex_t rhs) { *lhs = _mm_add_pd(*lhs, rhs); }
inline void subeq(complex_t* lhs, const complex_t rhs) { *lhs = _mm_sub_pd(*lhs, rhs); }
inline void muleq_scalar(complex_t* lhs, const Scalar rhs) { *lhs = _mm_mul_pd(*lhs, rhs); }
inline void muleq(complex_t* a_b, const complex_t c_d)
{
    complex_t a_a = _mm_permute_pd(*a_b, 0);
    complex_t b_b = _mm_permute_pd(*a_b, 3);
    complex_t d_c = _mm_permute_pd(c_d, 1);
    complex_t bc_bd = _mm_mul_pd(b_b, d_c);
    complex_t ad_ac = _mm_mul_pd(a_a, c_d);
    *a_b = _mm_addsub_pd(ad_ac, bc_bd);
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
	: num_rows(0), num_cols(0), num_values(0), values(NULL)
	{
	}
	ComplexMatrix(uint32_t rows, uint32_t cols)
	{
		//		printf("allocate %p\n", this);
		allocate(rows, cols);
	}
	~ComplexMatrix()
    {
//		printf("destroy %p\n", this);
		destroy();
    }
    void set_random_unitary()
    {
//		Returns a U(n) unitary of size n - random by the Haar measure
//			Taken from "How to generate random matrices from the classical compact groups" (Francesco Mezzadri)
//		Original Python:
//		z = (randn(n,n) + 1j*randn(n,n))/sqrt(2.0)
//			q,r = linalg.qr(z)
//			d = diagonal(r)
//			ph = d/absolute(d)
//			q = multiply(q,ph,q)
//			return matrix(q)
			
		double one_over_root_2 = 1.0 / sqrt(2.0);

		for (uint32_t i = 0; i < num_values; ++i)
        {
			double re = random_plus_minus() * one_over_root_2;
			double im = random_plus_minus() * one_over_root_2;
			values[i] = to_complex(re, im);
        }
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
	void allocate(uint32_t rows, uint32_t cols)
	{
		num_rows = rows;
		num_cols = cols;
		num_values = num_rows * num_cols;
#ifdef MSVC
		values = (complex_t*)_aligned_malloc(num_values * sizeof(complex_t), 16);
#else
		values = (complex_t*)memalign(16, num_values * sizeof(complex_t));
#endif
	}
    void mag_sqr(RealMatrix& dst) const;
    void debug_print() const;

private:
    void destroy()
    {
#ifdef MSVC
		if (values)
			_aligned_free(values);
#else
		if (values)
			free(values);
#endif
		values = NULL;
	}
    uint32_t num_rows;
    uint32_t num_cols;
    uint32_t num_values;
    complex_t* values;
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

