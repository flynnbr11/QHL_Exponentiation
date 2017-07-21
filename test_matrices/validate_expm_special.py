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


size=1
#mtx= rm.get_matrix(size)
identity = np.array([[1+0j, 0+0j], [0+0j, 1+0j]])
#mtx=identity

mtx = np.array([[1+0j, 2+0j], [3+0j, 4+0j]])

linalg_output=scipy.linalg.expm(mtx)
sparse_linalg_output = scipy.sparse.csc_matrix(linalg_output)

sparse_mtx = scipy.sparse.csc_matrix(mtx)
scipy_output = scipy.sparse.linalg.expm(sparse_mtx)


expm_special_output = mu.expm_special(mtx)
sparse_expm_special_output = scipy.sparse.csc_matrix(expm_special_output)

print 'Matrix:'
print mtx

print '----- Exponentiated ----- '
print 'Linalg:'
print 'Full output:' 
print linalg_output
print 'Output converted to sparse:'
print sparse_linalg_output

print 'Scipy:'
print scipy_output


print '---- ExpmSpecial ---- :'
print 'Before sparsing:'
print expm_special_output
print 'After sparsing:'
print sparse_expm_special_output

if np.all(expm_special_output == scipy_output):
	print 'Correct: Expm gives identical output to linalg.'
else: 
	print 'Incorrect: Expm does not give same output as linalg.'

