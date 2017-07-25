
import numpy as np
import libmatrix_utils as libmu
def test_run():
    print 'hi!'
    libmu.simple_test()

def expm_special(src, precision=1e-3):
    dst = np.ndarray(shape=(np.shape(src)[0], np.shape(src)[1]), dtype=np.complex128)
    result = libmu.expm_special_cpp(src, dst, precision)
    print result
    return dst
