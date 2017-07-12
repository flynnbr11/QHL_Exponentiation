// Copyright 2017 University of Bristol

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include "python_interface.h"
#include "matrix_utils.h"

#ifdef PLATFORM_OSX
# include <Python/Python.h>
#else
# include <Python.h>
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

std::string test_expm()
{
    uint32_t matrix_size = 8;
    ComplexMatrix mat(matrix_size, matrix_size);

    // Make it a pretend sparse hermitian matrix
    complex_t zero = to_complex(0.0, 0.0);
    complex_t one = to_complex(1.0, 0.0);
    for (uint32_t row = 0; row < matrix_size; ++row)
    {
        complex_t* prow = mat.get_row(row);
        for (uint32_t col = row; col < matrix_size; ++col)
        {
            complex_t* pcol = mat.get_row(col);
            if (row == col)
                prow[col] = pcol[row] = one;
            else
                prow[col] = pcol[row] = zero;
        }
    }
    // Add an X gate
    mat[1][1] = zero;
    mat[3][3] = zero;
    mat[1][3] = one;
    mat[3][1] = one;


    mat.debug_print();
    return "ok";
}


static PyObject* SimpleTest(PyObject *self, PyObject *args)
{
    std::string result_str = test_expm();
    return Py_BuildValue("s", result_str.c_str());
}

static PyObject* ExpmSpecial(PyObject *self, PyObject *args)
{
    std::string result_str;
    float precision = 0.0f;
    PyArrayObject* src_matrix;
    PyArrayObject* dst_matrix;

    if (!PyArg_ParseTuple(args, "O!O!f", &PyArray_Type, &src_matrix, &PyArray_Type, &dst_matrix, &precision))
    {
        fprintf(stderr, "Error: expm_special() arguments don't match, at %s %s:%d\n", __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }

    size_t src_rows = (size_t)PyArray_DIM(src_matrix, 0);
    size_t src_cols = (size_t)PyArray_DIM(src_matrix, 1);
    size_t dst_rows = (size_t)PyArray_DIM(dst_matrix, 0);
    size_t dst_cols = (size_t)PyArray_DIM(dst_matrix, 1);
    size_t src_stride = PyArray_STRIDES(src_matrix)[0];
    size_t dst_stride = PyArray_STRIDES(dst_matrix)[0];
    bool   src_is_complex = PyArray_ISCOMPLEX(src_matrix);
    bool   dst_is_complex = PyArray_ISCOMPLEX(dst_matrix);
    const complex_t* src_ptr = (const complex_t*)PyArray_DATA(src_matrix);
    complex_t* dst_ptr = (complex_t*)PyArray_DATA(dst_matrix);
    const char* error_str = NULL;

    // Check the matrces
    if (src_rows != src_cols || dst_rows != dst_cols)
        error_str = "Src and dest matrices must be square";
    if (src_rows != dst_rows || src_cols != dst_cols)
        error_str = "Src and dest matrices must be the same size";
    if (!src_is_complex || !dst_is_complex)
        error_str = "Src and dest matrices must be complex double-precision float";
    if (src_stride != src_cols * sizeof(complex_t) || dst_stride != dst_cols * sizeof(complex_t))
        error_str = "Unexpected stride length; matrices may not be packed complex double";
    if (error_str)
    {
        fprintf(stderr, "Error: %s, at %s %s:%d\n", error_str, __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }


    for (size_t i = 0; i < dst_rows * dst_cols; ++i)
        dst_ptr[i] = to_complex((double)i, 0.1);


    result_str = "ok";
    return Py_BuildValue("s", result_str.c_str());
}

static PyMethodDef matrix_utils_methods[] = {
    {"simple_test",             SimpleTest,                 METH_VARARGS, "Just a test."},
    {"expm_special_cpp",        ExpmSpecial,                METH_VARARGS, "Exponentiate a diagonal sparse matrix."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyObject *theError;
PyMODINIT_FUNC initlibmatrix_utils(void)
{
    PyObject *m;

    m = Py_InitModule("libmatrix_utils", matrix_utils_methods);
    if (m == NULL)
        return;

    import_array();

    theError = PyErr_NewException((char*)"libmatrix_utils.error", NULL, NULL);
    Py_INCREF(theError);
    PyModule_AddObject(m, "error", theError);
}



