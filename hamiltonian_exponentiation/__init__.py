""" 
Functions which are callable once hamiltonian_exponentiation has been imported.
- exp_ham: computes exp^{iHt} or exp^{-iHt}.
- random_hamiltonian: returns a Hamiltonian randomly generated by Pauli matrices. 
"""
from __future__ import print_function # so print doesn't show brackets
from inspect import currentframe


def unitary_evolve(ham, t, input_probe, use_sparse_dot_function=False, precision = 1e-6, k_max=None, plus_or_minus = -1.0, precision_cpp=1e-18, scalar_cutoff = 10, print_method=False, enable_sparse_functionality = True, sparse_min_qubit_number = 7):
    
    """
    Pass ham, t, input_probe (i.e. state) into this function
    It will send to exponentiation function, and perform matrix-vector multiplication of the exponentiated 
    Hamiltonian with the given state. 
    Optional arguments are the same as in exp_ham, and are passed directly to it. 
    """
    
    
    if not use_sparse_dot_function: 
      import numpy as np
      return np.dot(
                  exp_ham(ham, t, precision, k_max, plus_or_minus, precision_cpp, scalar_cutoff, print_method, enable_sparse_functionality, sparse_min_qubit_number),
                  input_probe)
      
    else: 
      from scipy.sparse import csr_matrix
      return csr_matrix(
                  exp_ham(ham, t, plus_or_minus, precision_cpp, scalar_cutoff, print_method, enable_sparse_functionality, sparse_min_qubit_number)
              ).dot(input_probe)


def exp_ham(src, t, precision=1e-6, k_max=None, plus_or_minus = -1.0, precision_cpp=1e-18, scalar_cutoff = 10, print_method=False, enable_sparse_functionality = True, sparse_min_qubit_number = 7):
    import numpy as np
    if np.shape(src)[0] != np.shape(src)[1]: 
        ("Hamiltonian: ", src)
        raise("Expected Square matrix, given matrix of shape ", np.shape(src))
    elif src.ndim != 2:
        print("Hamiltonian: ", src)
        print("Expected src with two dimensions; got ", src.ndim)
    n_qubits=np.log2(np.shape(src)[0])

    if src.dtype != 'complex128':
        # Ensure datatype is complex
        src = src.astype('complex128')

    if k_max is None:
        k_max = k_max_from_precision(precision)

    
    if n_qubits >= sparse_min_qubit_number and enable_sparse_functionality:
      return np.array(exp_ham_sparse(src, t, k_max, plus_or_minus = plus_or_minus, precision_cpp=precision_cpp, scalar_cutoff=scalar_cutoff, print_method=print_method))
    
    else: 
      return np.array(exponentiate_ham(src, t, k_max, plus_or_minus = plus_or_minus, precision_cpp=precision_cpp, scalar_cutoff=scalar_cutoff, print_method=print_method))


def exp_ham_trotter(src, t, plus_or_minus = -1.0, precision_cpp=1e-18, scalar_cutoff = 10, print_method=False, trotterize_by=1.0):
    import numpy as np
    from numpy import linalg as nplg
    n_qubits=np.log2(np.shape(src)[0])
    n = trotterize_by
    
    if n_qubits >= 7:
      exp_iHt = exp_ham_sparse(src, t, plus_or_minus = plus_or_minus, precision_cpp=precision_cpp, scalar_cutoff=scalar_cutoff, print_method=print_method)
  
    else:     
      if n == 1.0: 
        if print_method: 
          print ("Not Trotterized")
        exp_iHt = exponentiate_ham(src, t, precision_cpp=precision_cpp, plus_or_minus=plus_or_minus, scalar_cutoff=scalar_cutoff, print_method=print_method)
      else: 
        if print_method: 
          print("Trotterized; trot = ", n)
        time_over_n = t/trotterize_by
        exp_iHt_over_n = exponentiate_ham(src, time_over_n, precision_cpp=precision_cpp, scalar_cutoff=scalar_cutoff, print_method=print_method)
        exp_iHt = nplg.matrix_power(exp_iHt_over_n, n)
    return exp_iHt

def exponentiate_ham(src, t, k_max=40, plus_or_minus = -1.0, precision_cpp=1e-18, scalar_cutoff = 10, print_method=False):
    """
    Calls on C++ function to compute e^{-iHt}.
    Provide parameters: 
    - src: Hamiltonian to exponentiate
    - t: time
    - plus_or_mins: 1.0 to compute e^{iHt}; -1.0 to compute e^{-iHt}. Default -1.
    - precision_cpp: when matrix elements are changed by this amount or smaller, exponenitation is truncated.
    """
    import libmatrix_utils as libmu
    import numpy as np
    n_qubits = np.log2(np.shape(src)[0])
    max_element = np.max(np.abs(src))
    """
    if max_element == 0.0:
      max_element = 1.0
    
    python_scalar = max_element * t
    #print("Python scalar=", python_scalar)

    test_new_case = False

    if test_new_case == False:
        if max_element <= 1.0:

            new_src = src
            scalar = t
        else:
            new_src = src/max_element
            scalar = max_element * t
    else:

        new_src = src
        scalar = t
    """

    new_src = src
    scalar = t

    if(print_method):
        print("Not Sparse function. N qubits = ", np.log2(np.shape(src)[0]))
        print("scalar = ", scalar, " \t cutoff = ", scalar_cutoff)
    if(scalar > scalar_cutoff):
      #print("Scalar > Scalar cutoff. Scalar:", scalar, "\t Cutoff:", scalar_cutoff) 
    # If matrix scalar is larger than the defined cutoff for known accuracy, default to using linalg.
      import scipy
      from scipy import linalg
      if(print_method):
        print("Time = ", t, "\t element = ", max_element, "\t Scalar = ", scalar, " \t Linalg (scalar).")
      if(n_qubits >= 6 ): # Large matrices -- worth using sparse.linalg
        from scipy import sparse
        from scipy.sparse import linalg
        from scipy.sparse import csc_matrix
        expd_mtx = (scipy.sparse.linalg.expm(scipy.sparse.csc_matrix(plus_or_minus*1.j*src*t))).todense()      
      else: 
        expd_mtx = scipy.linalg.expm(plus_or_minus*1.j*src*t)
      return expd_mtx
    else:
      dst = np.ndarray(shape=(np.shape(src)[0], np.shape(src)[1]), dtype=np.complex128)
      
      
#      k_max = 22
      inf_reached = libmu.exp_pm_ham(new_src, dst, plus_or_minus, scalar, precision_cpp, k_max) # Call to C++ custom exponentiation function
      if(inf_reached):
        #print("Note: C++ function diverged; using linalg.")
        if(print_method):
          print("inf reached = ", inf_reached)
        import scipy
        from scipy import linalg
        if(n_qubits >= 6): # Large matrices -- worth using sparse.linalg
          from scipy import sparse
          from scipy.sparse import linalg
          from scipy.sparse import csc_matrix
          expd_mtx = (scipy.sparse.linalg.expm(scipy.sparse.csc_matrix(plus_or_minus*1.j*src*t))).todense()      
        else: 
          expd_mtx = scipy.linalg.expm(plus_or_minus*1.j*src*t)
          if(print_method):
            print("Time = ", t, "\t element = ", max_element, "\t Scalar = ", scalar, " \t Linalg (inf).")
        return expd_mtx
      else:
        if(print_method):
          print("Time = ", t, "\t element = ", max_element, "\t Scalar = ", scalar, " \t Custom.")
        return dst


def get_linenumber():
    cf = currentframe()
    return cf.f_back.f_lineno


def exp_ham_sparse(src, t, k_max=40, plus_or_minus = -1.0, precision_cpp=1e-18, scalar_cutoff = 10, print_method=False, trotterize_by=1.0):
  import numpy as np
  import libmatrix_utils as libmu
  max_element = np.max(np.abs(src))
  if max_element == 0.0:
    max_element = 1.0
  n_qubits = np.log2(np.shape(src)[0])
  new_src = src/max_element
  scalar = max_element * t
  if(print_method):
      print("Sparse function used. N qubits = ", np.log2(np.shape(src)[0]))

  if(scalar > scalar_cutoff):
    import scipy
    from scipy import linalg
    if(print_method):
      print("Time = ", t, "\t element = ", max_element, "\t Scalar = ", scalar, " \t Linalg (scalar).")
    if(n_qubits >= 6): # Large matrices -- worth using sparse.linalg
      from scipy import sparse
      from scipy.sparse import linalg
      from scipy.sparse import csc_matrix
      sparse_mtx = scipy.sparse.linalg.expm(scipy.sparse.csc_matrix(plus_or_minus*1.j*src*t))
      return sparse_mtx.todense()
    else: 
      return scipy.linalg.expm(plus_or_minus*1.j*src*t)
  else: 
    max_nnz_in_any_row, num_nnz_by_row, nnz_col_locations, nnz_vals  = matrix_preprocessing(new_src)
    dst = np.ndarray(shape=(np.shape(src)[0], np.shape(src)[1]), dtype=np.complex128)

    # Call C++ extension
   # k_max = 18    
    inf_reached = libmu.exp_pm_ham_sparse(dst, nnz_vals, nnz_col_locations, num_nnz_by_row, max_nnz_in_any_row, plus_or_minus, scalar, precision_cpp, k_max)

    if(inf_reached):
      del dst
      import scipy
      from scipy import linalg
      if(n_qubits >= 6): # Large matrices -- worth using sparse.linalg
        from scipy import sparse
        from scipy.sparse import linalg
        from scipy.sparse import csc_matrix
        sparse_mtx = scipy.sparse.linalg.expm(scipy.sparse.csc_matrix(plus_or_minus*1.j*src*t))
        return sparse_mtx.todense()      
      else: 
        if(print_method):
          print("Time = ", t, "\t element = ", max_element, "\t Scalar = ", scalar, " \t Linalg (inf).")
        return scipy.linalg.expm(plus_or_minus*1.j*src*t)
    else:
      if(print_method):
        print("Time = ", t, "\t element = ", max_element, "\t Scalar = ", scalar, " \t Custom.")
      return dst


def matrix_preprocessing(ham): 
    import scipy
    from scipy.sparse import csr_matrix
    import numpy as np
    sp_ham = scipy.sparse.csr_matrix(ham)
    num_rows = np.shape(ham)[0]
    num_nnz_by_row = sp_ham.getnnz(1)
    max_nnz_in_any_row = max(num_nnz_by_row)
    nnz_vals = np.zeros((num_rows, max_nnz_in_any_row), dtype=np.complex128)
    nnz_col_locations = np.zeros((num_rows, max_nnz_in_any_row), dtype=int)
    data = sp_ham.data
    col_locations = sp_ham.indices

    k=0
    for i in range(num_rows):
        for j in range(num_nnz_by_row[i]):
            nnz_vals[i][j] = data[k]
            nnz_col_locations[i][j] = col_locations[k]
            k+=1

    return max_nnz_in_any_row, num_nnz_by_row, nnz_col_locations, nnz_vals 


def random_state(num_qubits):
    import numpy as np
    dim = 2**num_qubits
    real = np.random.rand(1,dim)
    imaginary = np.random.rand(1,dim)
    complex_vectors = np.empty([1, dim])
    complex_vectors = real +1.j*imaginary
    norm_factor = np.linalg.norm(complex_vectors)
    probe = complex_vectors/norm_factor
    return probe[0][:]


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
  return output
  


def k_max_from_precision(precision):
    if (precision <= 1e-14):
        k_max = 20
    elif (1e-14 < precision <= 1e-12):     
        k_max = 25
    elif (1e-12 < precision <= 1e-9):     
        k_max = 30
    elif (1e-9 < precision <= 1e-8):     
        k_max = 35
    elif (1e-8 < precision <= 1e-6):     
        k_max = 40
    elif (1e-6 < precision <= 1e-4):     
        k_max = 45
    elif (1e-4 < precision <= 1e-2):     
        k_max = 50
    else:
        k_max = 40 # default corresponds to precision 10^{-6}
        
        
    print("For precision", precision, "k_max=", k_max)
    return k_max
    
    

  
def test_exp_ham_function(
    min_qubit = 1,
    max_qubit = 9,
    test_large_cases = False, # default is not to compute linalg.expm for num_qubit > 9.
    threshold = 1e-14, # Tolerance at which matrix elements considered equal
    check_all_equal = True,
    print_times_and_ratios = False,
    print_exp_method = False, 
    test_sparse_speedup = False,
    num_tests = 1
):
    """
    Test that the function behaves as expected by computing 
    exponentiated Hamiltonian via the function provided herein, exp_ham(ham,t)
    and comparing the result with that given by linalg.expm(-1j*ham*t),
    accurate to the threshold specified when calling this function.
    """
    import random
    import time
    from scipy import linalg
    import numpy as np
    
    for k in range(num_tests):
        print("\n\nTest ", k)
        for num_qubit in range(min_qubit, max_qubit+1):
            print("\n", num_qubit, " qubits." )
            ham = random_hamiltonian(num_qubit)
            t = random.random()
            c1 = time.time()
            custom = exp_ham(ham, t, k_max=40, print_method=print_exp_method)
            c2 = time.time()
            tc = c2-c1
            if print_times_and_ratios: print("sparse takes ", tc, " seconds.")
            
            if(num_qubit >= 7 and test_sparse_speedup == True):
                d1 = time.time()
                custom_without_sparse = exp_ham(ham, t, enable_sparse_functionality=False)
                d2 = time.time()
                td=d2 - d1
                if print_times_and_ratios: print("non-sparse takes ", td, " seconds.")
            
            if num_qubit < 10 or test_large_cases is True:
                l1=time.time()
                lin = linalg.expm(-1j*ham*t)
                l2=time.time()
                tl = l2 - l1
                if print_times_and_ratios: print("linalg takes ", tl, " seconds.")
                
                if not np.all( np.abs(lin - custom) < threshold):
                    print("Not equal")
                    check_all_equal = False
                    
              
                if print_times_and_ratios: print("time ratio linalg:custom", (tl/tc)  )

            if num_qubit >= 7 and test_sparse_speedup == True:
                if print_times_and_ratios: print("time ratio sparse:non-sparse : ", (td/tc)  )
            
    num_models = (max_qubit - min_qubit + 1) * num_tests
    
    print("\n\nChecked ", num_models, " random Hamiltonians.")
    if check_all_equal is not True:
        print("---- ---- PROBLEM: Not all matrix elements equal ---- ----")
    else:
        print("---- ---- All matrix elements equal ---- ----")
  
def get_linenumber():
    """
    For use internally for debugging etc.
    """
    cf = currentframe()
    return cf.f_back.f_lineno
  
  
  
  
