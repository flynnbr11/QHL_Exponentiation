import numpy as np
import scipy as sp
from scipy import linalg 
import time as time
import matrix_utils as mu
import random_matrix as rm
import matplotlib.pyplot as plt

# matrix_size: how many qubits. matrix will be square of 2^size X 2^size
do_loop=1 ## turn off to examine calls as below (under else: )

if do_loop==1:
	matrix_sizes = [1,2,3]
	track_times=np.zeros([len(matrix_sizes), 4])
	
	for i in range(0,len(matrix_sizes)):
		size=matrix_sizes[i]
		mat= rm.get_matrix(size)
		
		a = time.time()
		linalg.expm(mat)
		b = time.time()

		c=time.time()
		emat = mu.expm_special(mat)
		d=time.time()

		time_linalg = b-a
		time_expm = d-c

		track_times[i,0]= matrix_sizes[i]
		track_times[i,1]= time_linalg
		track_times[i,2]= time_expm
		track_times[i,3]= time_linalg/time_expm

	print('tracking:')
	print(track_times)
	
	
else: 	
	#mat = np.identity(matrix_size, dtype=np.complex128)
	matrix_size = 1
	test_matrix = rm.get_matrix(matrix_size)

	print('Matrix of size 2^', matrix_size )
	print('original matrix:')
	print(test_matrix)

	lin_mat = linalg.expm(test_matrix)
	exp_mat = mu.expm_special(test_matrix)

	diff = lin_mat - exp_mat

	print('linalg:')
	print(lin_mat)
	print('expm:')
	print(exp_mat)
	print('difference between linalg and expm fnc:')
	print(diff)
