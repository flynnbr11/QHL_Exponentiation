import hamiltonian_exponentiation as h
import numpy as np
from scipy import linalg
from scipy import sparse
import time as time 

def time_hamilt(size):
	mtx = h.random_hamiltonian(size)

	ham_time_start = time.time()
	ham = h.expn_hamilt(mtx)
	ham_time_end = time.time()

	ham_time = ham_time_end - ham_time_start

	#print("Times for matrix of n qubits. n=", size)
	#print("\t HamExp : ", ham_time)

	print("Number qubits = ", size, "\t Time = ", ham_time) 



size = 10
tol = 1e-15
test_comparison = 0

qubit_min = 5
qubit_max = 7

for size in range(qubit_min, qubit_max+1):
	time_hamilt(size)




if test_comparison == 1:
	lin_time_start = time.time()
	lin = linalg.expm(mtx)
	lin_time_end = time.time()
	lin_time = lin_time_end - lin_time_start

	diff = lin - ham 
	largest_element = 0

	for i in range(np.shape(diff)[0]):
		for j in range(np.shape(diff)[1]):
			if diff[i][j] > largest_element:
				largest_element = diff[i][j]
		

	print("Largest difference in elements: ", largest_element)
	print("Times for matrix of n qubits. n=", size)
	print("\t LinAlg : ", lin_time)
	print("\t HamExp : ", ham_time)
	

