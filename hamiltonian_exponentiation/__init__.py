
import numpy as np
import libmatrix_utils as libmu
import future
import past
"""
def test_run():
    print 'hi!'
    libmu.simple_test()
"""
def expn_hamilt(src, precision=1e-18):
    dst = np.ndarray(shape=(np.shape(src)[0], np.shape(src)[1]), dtype=np.complex128)
    result = libmu.expm_special_cpp(src, dst, precision)
    print(result)
    return dst
    
def test_installation():
		print("Installation successful -- Expn Hamiltonian.")
		

