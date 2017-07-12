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

void ComplexMatrix::mag_sqr(RealMatrix& dst) const
{
    const complex_t* psrc = get_row(0);
    double* pdst = dst.get_row(0);
    for (uint32_t i = 0; i < num_values; ++i)
        *pdst++ = ::mag_sqr(*psrc++);
}



void ComplexMatrix::expm_special(ComplexMatrix& dst, double precision) const
{
    printf("precision = %f\n", precision);
    for (size_t i = 0; i < num_rows * num_rows; ++i)
        dst.values[i] = add(values[i], values[i]);
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
    Scalar half = to_scalar(0.5);
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
        Scalar si = to_scalar(-1.0 + 2.0 * bit_get(yi, zi));

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
}
