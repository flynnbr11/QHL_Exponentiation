# import numpy as np
# import libmatrix_utils as libmu 

def exp_minus_i_h_t(src, t, precision=1e-16):
    """
		Calls on C++ function to compute e^{-iHt}.
		Provide parameters: 
		- src: Hamiltonian to exponentiate
		- t: time
		- precision: when matrix elements are changed by this amount or smaller, exponenitation is truncated.
		"""
    import libmatrix_utils as libmu
    import numpy as np
    dst = np.ndarray(shape=(np.shape(src)[0], np.shape(src)[1]), dtype=np.complex128)
    inf_reached = libmu.e_minus_i_h_t(src, dst, t, precision)
#    print(inf_reached)
    if(inf_reached):
#      print("Incorrect - reverting to linalg sparse function.")
      import scipy
      from scipy import sparse
      from scipy import linalg
      from scipy.sparse.linalg import LinearOperator
      from scipy.sparse import csr_matrix
      expd_mtx = sparse.linalg.expm(-1.j*src*t)			
      return expd_mtx
    else:
#      print("Correct:")
      return dst

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
