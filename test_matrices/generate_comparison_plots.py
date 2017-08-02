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

# matrix_size: how many qubits. matrix will be square of 2^size X 2^size
#matrix_sizes = [1,2,3]
high=11 # max number of qubits to go to
low=1 # Min number of qubits to simulate to
matrix_sizes = range(low,high+1)
track_times=np.zeros([len(matrix_sizes), 6])

time_before_loop = time.time()

linalg_flag = 1 

for i in range(0,len(matrix_sizes)):
	size=matrix_sizes[i]
	mtx= rm.get_matrix(size)
	
	a=0
	b=0

	if linalg_flag == 0:	
		a = time.time()
		linalg.expm(mtx)
		b = time.time()

	sparse_mtx = scipy.sparse.csc_matrix(mtx)
	e = time.time()
	scipy.sparse.linalg.expm(sparse_mtx)
	f = time.time()


	c=time.time()
	mu.expm_special(mtx)
	d=time.time()

	time_linalg = b-a
	time_expm = d-c
	time_sparse_linalg=f-e

	track_times[i,0]= size
	if linalg_flag == 0:
		track_times[i,1]= time_linalg
		track_times[i,4]= time_linalg/time_expm
	else:
		track_times[i,1]= None
		track_times[i,4]= None
		
	track_times[i,2]= time_sparse_linalg
	track_times[i,3] = time_expm
	track_times[i,5]= time_sparse_linalg/time_expm
	
	if time_linalg > 15.0:
		linalg_flag = 1
			
plt.plot(basex=2)
plt.semilogy(track_times[:,0], track_times[:,1], label='LinAlg', marker='o')
plt.semilogy(track_times[:,0], track_times[:,2], label='Sparse LinAlg', marker='o')
plt.semilogy(track_times[:,0], track_times[:,3], label='ExpmSpecial', marker='o')
plt.axhline(y=1, color='black', label='1 second')
plt.title('Times for expm functions up to '+str(high)+'qubits')
plt.xticks(np.arange(min(track_times[:,0]), max(track_times[:,0])+1, 1))
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend()
plt.savefig('sparse_expm_comparison_plots/opt4_times_'+str(low)+'_to_'+str(high)+'_qubits.png')

plt.clf()
plt.plot(basex=2)
plt.plot(track_times[:,0], track_times[:,4], label='Ratio improvement Vs Linalg', marker='o')
plt.plot(track_times[:,0], track_times[:,5], label='Ratio improvement Vs Sparse LinAlg', marker='o')
plt.axhline(y=1, color='black', label='Improvement/Loss Cutoff')
plt.title('ExpmSpecial Improvement up to '+str(high)+'qubits')
plt.xticks(np.arange(min(track_times[:,0]), max(track_times[:,0])+1, 1))
plt.xlabel('Number of Qubits')
plt.ylabel('Improvement')
plt.legend()
plt.savefig('sparse_expm_comparison_plots/opt4_improvement_'+str(low)+'_to_'+str(high)+'_qubits.png')

np.savetxt('timings/opt4_times_'+str(low)+'_to_'+str(high)+'_qubits.csv', track_times, delimiter=",")

time_after_loop = time.time()
time_taken = time_after_loop-time_before_loop
print 'Time to do all calculations: ', time_taken

