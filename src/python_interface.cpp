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

const bool PRINT_LINE=false;

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
    return "ok: test_expm";
}


static PyObject* SimpleTest(PyObject *self, PyObject *args)
{
    std::string result_str = test_expm();
    return Py_BuildValue("s", result_str.c_str());
}


static PyObject* Exp_iHt_sparse(PyObject *self, PyObject *args)
{
    //printf("Python interface for sparse function reached.\n");
    std::string result_str;
    double precision = 0.0f;
    double scale = 0.0f;
    double plus_or_minus = 0.0f;
    double max_nnz_in_any_row = 0.0f;
    PyArrayObject* nnz_vals_p;
    PyArrayObject* nnz_col_locations_p;
    PyArrayObject* num_nnz_by_row_p;
    PyArrayObject* dst_matrix;

//  libmu.exp_pm_ham_sparse(dst, nnz_valz, nnz_col_locations, num_nnz_by_row, max_nnz_in_any_row, plus_or_minus, scalar, precision)

//    if (!PyArg_ParseTuple(args, "O!O!ddd", &PyArray_Type, &src_matrix, &PyArray_Type, &dst_matrix, &plus_or_minus, &scale, &precision))


     if (!PyArg_ParseTuple(args, "O!O!O!O!dddd",  &PyArray_Type, &dst_matrix, &PyArray_Type, &nnz_vals_p, &PyArray_Type, &nnz_col_locations_p, &PyArray_Type, &num_nnz_by_row_p, &max_nnz_in_any_row , &plus_or_minus, &scale, &precision))
    {
        fprintf(stderr, "Error: Sparse function arguments don't match, at %s %s:%d\n", __FUNCTION__, __FILE__, __LINE__);
        return NULL;
    }
    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);

    size_t dst_rows = (size_t)PyArray_DIM(dst_matrix, 0);
    size_t dst_cols = (size_t)PyArray_DIM(dst_matrix, 1);


    uint32_t max_nnz = (uint32_t) max_nnz_in_any_row;
    uint32_t num_rows = (uint32_t) dst_rows;



    uint32_t* nnz_by_row = (uint32_t*)PyArray_DATA(num_nnz_by_row_p);;
    complex_t* tmp_nnz_vals = (complex_t*)PyArray_DATA(nnz_vals_p);
    uint32_t* tmp_col = (uint32_t*)PyArray_DATA(nnz_col_locations_p);

    uint32_t col_rows = (uint32_t)PyArray_DIM(nnz_col_locations_p, 0);
    uint32_t cols_cols = (uint32_t)PyArray_DIM(nnz_col_locations_p, 1);

    uint32_t* tst_array = (uint32_t*)PyArray_DATA(nnz_col_locations_p);
    complex_t* tst_nnz_array = (complex_t*)PyArray_DATA(nnz_vals_p);

    uint32_t** nnz_col_locations;
    complex_t** nnz_vals;
    nnz_vals = new complex_t*[num_rows];
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
      	nnz_col_locations[i][j] = tmp_col[2*(i*max_nnz+j)];
        nnz_vals[i][j] = tmp_nnz_vals[i*max_nnz+j];
      }
    }

    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);

    bool plus_minus_flag = false; // if flag is false, e^{-iHt}; if flag is true, e^{iHt}
    if (plus_or_minus == 1.0) 
    {
      plus_minus_flag = true;
    } // else plus_minus_flag=false

    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
    
    ComplexMatrix hamiltonian(num_rows, max_nnz, nnz_by_row, nnz_col_locations, nnz_vals);
    complex_t* dst_ptr = (complex_t*)PyArray_DATA(dst_matrix);
    bool exp_reached_inf = 0;

    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
    
    exp_reached_inf = hamiltonian.exp_ham_sparse(dst_ptr, scale, precision, plus_minus_flag);
    
    /*
    Deleting pointers to try eliminate memory leak in sparse function.
    */

    //*    
    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
    for(uint32_t i=0; i<num_rows; i++)
    {
    if(PRINT_LINE) fprintf(stderr,"%d\n", i);
    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
      delete[] nnz_vals[i];
    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
      delete[] nnz_col_locations[i];
    }
    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
    delete[] nnz_vals;
    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
    delete[] nnz_col_locations;
    if(PRINT_LINE) fprintf(stderr,"Running at :%d\n", __LINE__);
    //*/
    
    /*
    Py_DECREF(nnz_vals_p);    
    Py_DECREF(nnz_col_locations_p);    
    Py_DECREF(num_nnz_by_row_p);    
    //Py_DECREF(dst_matrix);    
    //*/
    //printf("End of sparse Python/C++ interface\n");
    return Py_BuildValue("b", exp_reached_inf);
}




static PyObject* Exp_iHt(PyObject *self, PyObject *args)
{
    std::string result_str;
    double precision = 0.0f;
    double scale = 0.0f;
    double plus_or_minus = 0.0f;
    PyArrayObject* src_matrix;
    PyArrayObject* dst_matrix;

    // libmu.exp_pm_ham(new_src, dst, plus_or_minus, scalar, precision)
    if (!PyArg_ParseTuple(args, "O!O!ddd", &PyArray_Type, &src_matrix, &PyArray_Type, &dst_matrix, &plus_or_minus, &scale, &precision))
    {
        fprintf(stderr, "Error: expm_minus_i_h_t() arguments don't match, at %s %s:%d\n", __FUNCTION__, __FILE__, __LINE__);
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
    complex_t* src_ptr = (complex_t*)PyArray_DATA(src_matrix);
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
    
    bool plus_minus_flag = false; // if flag is false, e^{-iHt}; if flag is true, e^{iHt}
    if (plus_or_minus == 1.0) 
    {
      plus_minus_flag = true;
    } // else plus_minus_flag=false

    const ComplexMatrix src(src_rows, src_cols, src_ptr);
    ComplexMatrix dst(dst_rows, dst_cols, dst_ptr);
	bool exp_reached_inf = false;
	exp_reached_inf = src.exp_ham(dst, scale, precision, plus_minus_flag);
	//	src.expm_minus_i_h_t(dst, time, precision, plus_minus_flag);
    
    result_str = "ok: e^{-iHt}";
    return Py_BuildValue("b", exp_reached_inf);
}


static PyMethodDef matrix_utils_methods[] = {
      {"exp_pm_ham_sparse",       Exp_iHt_sparse,      METH_VARARGS, "Use sparse functionality to exponentiate a Hamiltonian."},
	  	{"exp_pm_ham", 					    Exp_iHt,              METH_VARARGS, "Exponentiate {iHt} where H is input Hamiltonian, t is time given."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static PyObject *theError;

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC initlibmatrix_utils(void)
{
	int python_version = PY_MAJOR_VERSION;
	//printf("\n \n \n \n \n Python version : %d \n \n \n \n \n ", python_version);

    PyObject *m;

    m = Py_InitModule("libmatrix_utils", matrix_utils_methods);
    if (m == NULL)
        return;

    import_array();

    theError = PyErr_NewException((char*)"libmatrix_utils.error", NULL, NULL);
    Py_INCREF(theError);
    PyModule_AddObject(m, "error", theError);
}
//*/

/* Python3.5 C++ interface: Py_InitModule deprecated; use PyModuleDef and PyModule_Create instead */

//*
#elif PY_MAJOR_VERSION == 3
/* TODO: This section is reason updating function framework doesn't work for Python3+*/

static struct PyModuleDef libmatrix_utils =
{
	PyModuleDef_HEAD_INIT,
	"libmatrix_utils",
	"Matrix Utilities",
	-1,
	matrix_utils_methods
};

PyMODINIT_FUNC PyInit_libmatrix_utils(void)
{
    PyObject *m;

    m = PyModule_Create(&libmatrix_utils);
	
	if (m == NULL)
        return NULL;

    import_array();

    theError = PyErr_NewException((char*)"libmatrix_utils.error", NULL, NULL);
    Py_INCREF(theError);
    PyModule_AddObject(m, "error", theError);
    return m;
}
//*/

#endif
