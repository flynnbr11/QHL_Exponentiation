import scipy
from scipy import linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
import time as time
import matplotlib.pyplot as plt 
import numpy as np
import sys
import hamiltonian_exponentiation as ham_exp
import random_matrix as rm
np.set_printoptions(threshold=np.inf)

def check_correctness_expm_special(num_qubits=1, print_orig_matrix = False, print_full_outputs = False, print_sparse_outputs = False, print_difference = False):
	
	size=num_qubits
	mtx= rm.get_matrix(size)

	linalg_output=scipy.linalg.expm(mtx)
	#sparse_linalg_output = scipy.sparse.csc_matrix(linalg_output)

	#sparse_mtx = scipy.sparse.csc_matrix(mtx)
	#scipy_output = scipy.sparse.linalg.expm(sparse_mtx)

	expm_special_output = ham_exp.expn_hamilt(mtx)
	#sparse_expm_special_output = scipy.sparse.csc_matrix(expm_special_output)


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
		

	precision = 1e-15
	flag = 0 
	max_diff = 0
	"""
	print "Linalg : "
	print linalg_output
	print "Expm: "
	print expm_special_output
	"""
	rows = np.shape(mtx)[0]
	cols = np.shape(mtx)[1]
	for i in range(rows):
		for j in range(cols):
			diff = np.abs(expm_special_output[i][j] - linalg_output[i][j])
			if diff > max_diff:
				max_diff = diff

	return max_diff
	
"""
	if np.all(np.abs(np.real(expm_special_output) - np.real(scipy_output)) < 1e-16):
		#print 'Correct: Expm gives identical output to linalg.'
		return 0
	else: 
		diff_mtx = expm_special_output - linalg_output
		print diff_mtx
		#print 'Incorrect: Expm does not give same output as linalg.'
		return 1
"""	

check_sum =0
max_diff_all=0
for q in range(1, 12):
	for i in range(1, 10): 
		current = check_correctness_expm_special(num_qubits=q)
		if current > max_diff_all:
			max_diff_all = current
print "Largest difference: ", max_diff_all 
#			check_sum += check_correctness_expm_special(num_qubits=q)
# print 'Check sum = ', check_sum
