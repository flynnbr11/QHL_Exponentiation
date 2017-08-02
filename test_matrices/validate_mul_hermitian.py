import scipy
from scipy import linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
import time as time
import matplotlib.pyplot as plt 
import numpy as np
import sys
import matrix_utils as mu
import random_matrix as rm
np.set_printoptions(threshold=np.inf)

num_qubits=2

size=num_qubits
mtx1= rm.get_matrix(size)
mtx2= rm.get_matrix(size)
dst = np.identity(2**size, dtype=np.complex128)


expm_special_output = mu.expm_special(mtx1)
linalg_output=scipy.linalg.expm(mtx1)
#dst = mu.mul_hermitian(mtx2, mtx1)


print "Matrix: "
print mtx1
print "---- ----"
print "Exponentiated: "
print expm_special_output
print "---- ----"
print "Linalg: " 
print linalg_output
