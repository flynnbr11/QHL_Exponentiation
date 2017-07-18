import numpy as np
import sys
np.set_printoptions(threshold=np.inf)

def sigmaz():
    return np.array([[1+0j, 0+0j], [0+0j, -1+0j]])

def sigmax():
    return np.array([[0+0j, 1+0j], [1+0j, 0+0j]])

def sigmay():
    return np.array([[0+0j, -1j], [1j, 0+0j]])

def get_matrix(matrix_size):
	oplist = [sigmax(), sigmay(), sigmaz()]
	size = matrix_size
	select=np.round((len(oplist)-1)*np.random.rand(size))
	newoplist = [oplist[int(i)] for i in select]
	params=np.random.rand(size)
	
	if len(params)==1:
		output = params[0]*newoplist[0]
	else:			
		for i in range(len(params)-1):
				if i==0:
				    output = np.kron(params[i]*newoplist[i], params[i+1]*newoplist[i+1])
				else:
				    output = np.kron(output, params[i+1]*newoplist[i+1])
		output = np.reshape(output, [2**size,2**size])

	# print('Ratio of non zero to total elements:', np.count_nonzero(output), ':', ((2**size)**2))
	
	return output
