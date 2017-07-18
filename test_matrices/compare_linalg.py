import numpy as np
import scipy as sp
from scipy import linalg 
import time as time
import matrix_utils as mu
import random_matrix as rm
import matplotlib.pyplot as plt

# matrix_size: how many qubits. matrix will be square of 2^size X 2^size
#matrix_sizes = [1,2,3]
matrix_sizes = range(1,5)
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

	track_times[i,0]= 2**matrix_sizes[i]
	track_times[i,1]= time_linalg
	track_times[i,2]= time_expm
	track_times[i,3]= time_linalg/time_expm


plt.semilogx(basex=2)
plt.semilogy(track_times[:,0], track_times[:,1], label='LinAlg')
plt.semilogy(track_times[:,0], track_times[:,2], label='ExpmSpecial')
plt.xlabel('Square matrix size')
plt.ylabel('Time (seconds)')
plt.legend()
plt.savefig('times.png')

plt.clf()
plt.semilogx(basex=2)
plt.plot(track_times[:,0], track_times[:,3], label='Ratio improvement using ExpmSpecial')
plt.axhline(y=1, color='black', label='Improvement/Loss Cutoff')
plt.xlabel('Square matrix size')
plt.ylabel('Improvement')
plt.legend()
plt.savefig('improvement.png')
