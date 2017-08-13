# import numpy as np
# import libmatrix_utils as libmu 

def expn_hamilt(src, precision=1e-25):
    import libmatrix_utils as libmu
    import numpy as np
    dst = np.ndarray(shape=(np.shape(src)[0], np.shape(src)[1]), dtype=np.complex128)
    result = libmu.expm_special_cpp(src, dst, precision)
    print(result)
    return dst

def e_i_hamilt(src, precision=1e-25):
    import libmatrix_utils as libmu
    import numpy as np
    dst = np.ndarray(shape=(np.shape(src)[0], np.shape(src)[1]), dtype=np.complex128)
    result = libmu.exponentiate_i_hamiltonian(src, dst, precision)
    print(result)
    return dst

def test_install():
		print("Installation successful 15")	

def new_test_install():
		print("New Install test: 1")

"""
def sigmaz():
    import numpy as np
    return np.array([[1+0j, 0+0j], [0+0j, -1+0j]])

def sigmax():
		import numpy as np
    return np.array([[0+0j, 1+0j], [1+0j, 0+0j]])

def sigmay():
    import numpy as np
    return np.array([[0+0j, 0-1j], [0+1j, 0+0j]])
"""

def random_hamiltonian(number_qubits):
	"""
	Generate a random Hamiltonian - will be square with length/width= 2**number_qubits.
	Hamiltonian will be Hermitian so diagonally symmetrical elements are complex conjugate.
	Hamiltonian also formed by Pauli matrices.
	"""
	import numpy as np
	sigmax = np.array([[0+0j, 1+0j], [1+0j, 0+0j]])
	sigmay = np.array([[0+0j, 0-1j], [0+1j, 0+0j]])
	sigmaz = np.array([[1+0j, 0+0j], [0+0j, -1+0j]])
	
	oplist =  [sigmax, sigmay, sigmaz]
	# oplist = [sigmax(), sigmay(), sigmaz()]
	size = number_qubits
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
