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

def check_correctness_expm_special(num_qubits=1, print_orig_matrix = False, print_full_outputs = False, print_sparse_outputs = False, print_difference = False):
	
	size=num_qubits
	mtx= rm.get_matrix(size)

	# linalg_output=scipy.linalg.expm(mtx)
	# sparse_linalg_output = scipy.sparse.csc_matrix(linalg_output)

	sparse_mtx = scipy.sparse.csc_matrix(mtx)
	scipy_output = scipy.sparse.linalg.expm(sparse_mtx)

	expm_special_output = mu.expm_special(mtx)
	sparse_expm_special_output = scipy.sparse.csc_matrix(expm_special_output)


	if print_orig_matrix is True: 	
		print '---- ----'
		print 'Matrix:'
		print mtx


	if print_full_outputs is True:
		print '----- Linalg ----- '
		print linalg_output
		print '---- ExpmSpecial ---- :'
		print expm_special_output


	if print_sparse_outputs is True:
		print '----- Linalg ----- '
		print scipy_output

		print '----- Linalg Sparse ----- '
		print sparse_linalg_output


		print '----- Expm Special Sparse ----- '
		print sparse_expm_special_output

	if print_difference is True: 
		diff_mtx = expm_special_output - linalg_output
		print diff_mtx

	if np.all(expm_special_output - scipy_output < 1e-15):
		#print 'Correct: Expm gives identical output to linalg.'
		return 0
	else: 
		#print 'Incorrect: Expm does not give same output as linalg.'
		return 1
	
check_sum =0
for q in range(1, 5):
	for i in range(1, 5): 
		check_sum += check_correctness_expm_special(num_qubits=q)
print 'Check sum = ', check_sum
