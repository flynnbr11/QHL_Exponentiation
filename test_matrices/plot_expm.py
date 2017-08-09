import numpy as np
import scipy as sp
from scipy import linalg 
import time as time
import matrix_utils as mu
import random_matrix as rm
import matplotlib.pyplot as plt

# matrix_size: how many qubits. matrix will be square of 2^size X 2^size
#matrix_sizes = [1,2,3]
matrix_sizes = range(1,14)
track_times=np.zeros([len(matrix_sizes), 2])


for i in range(0,len(matrix_sizes)):
	size=matrix_sizes[i]
	mat= rm.get_matrix(size)
	
	c=time.time()
	mu.expm_special(mat)
	d=time.time()

	time_expm = d-c

	track_times[i,0]= 2**matrix_sizes[i]
	track_times[i,1]= time_expm


plt.semilogx(basex=2)
plt.semilogy(track_times[:,0], track_times[:,1], label='ExpmSpecial')
plt.xlabel('Square matrix size')
plt.ylabel('Time (seconds)')
plt.legend()
plt.savefig('expm_fncs_plots/expm_special_sparse_times.png')

