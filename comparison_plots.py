import scipy
from scipy import linalg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
import time as time
import matplotlib.pyplot as plt 
import numpy as np
import sys
import hamiltonian_exponentiation as h
np.set_printoptions(threshold=np.inf)

# matrix_size: how many qubits. matrix will be square of 2^size X 2^size
#matrix_sizes = [1,2,3]
high=5 # max number of qubits to go to
low=1 # Min number of qubits to simulate to
matrix_sizes = range(low,high+1)
track_times=np.zeros([len(matrix_sizes), 6])

time_before_loop = time.time()

linalg_flag = 1 
t=1
for i in range(0,len(matrix_sizes)):
	size=matrix_sizes[i]
#	mtx= rm.get_matrix(size)
	mtx = h.random_hamiltonian(size)
# TODO: Scale this up to large t*mtx
	
	a=0
	b=0

	if linalg_flag == 0:	
		a = time.time()
		linalg.expm(-1j*mtx*t)
		b = time.time()

	e = time.time()
	i_mtx = -1.j*mtx*t;
	sparse_mtx = scipy.sparse.csc_matrix(i_mtx)
	scipy.sparse.linalg.expm(sparse_mtx)
	f = time.time()

	c=time.time()
	h.exp_ham(mtx,t)
	d=time.time()

	g=time.time()
	h.exp_ham_sparse(mtx,t)
	g1=time.time()

	time_linalg = b-a
	time_expm = d-c
	time_sparse_linalg=f-e
	time_sparse_expm = g1-g

	track_times[i,0]= size
	if linalg_flag == 0:
		track_times[i,1]= time_linalg
	else:
		track_times[i,1]= None
		
	track_times[i,2]= time_sparse_linalg
	track_times[i,3] = time_expm
	track_times[i,3] = time_sparse_expm
	
	if time_linalg > 30.0:
		linalg_flag = 1
			
plt.plot(basex=2)
plt.semilogy(track_times[:,0], track_times[:,1], label='LinAlg', marker='o')
plt.semilogy(track_times[:,0], track_times[:,2], label='Sparse LinAlg', marker='o')
plt.semilogy(track_times[:,0], track_times[:,3], label='Custom Expm', marker='o')
plt.semilogy(track_times[:,0], track_times[:,4], label='Sparse Custom', marker='o')
plt.axhline(y=1, color='black', label='1 second')
plt.title('Times for expm functions up to '+str(high)+'qubits')
plt.xticks(np.arange(min(track_times[:,0]), max(track_times[:,0])+1, 1))
plt.xlabel('Number of Qubits')
plt.ylabel('Time (seconds)')
plt.legend()
plt.savefig('comparison_plots/times_'+str(low)+'_to_'+str(high)+'_qubits.png')

"""
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
plt.savefig('comparison_plots/improvement_'+str(low)+'_to_'+str(high)+'_qubits.png')

np.savetxt('timings/opt4_times_'+str(low)+'_to_'+str(high)+'_qubits.csv', track_times, delimiter=",")
"""
time_after_loop = time.time()
time_taken = time_after_loop-time_before_loop
print('Time to do all calculations: ', time_taken)

