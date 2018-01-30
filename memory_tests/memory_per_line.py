from __future__ import print_function # so print doesn't show brackets
import hamiltonian_exponentiation as h 
from psutil import virtual_memory

import sys, os        
import inspect
from memory_profiler import profile
from scipy import linalg
import sys

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno




print_mem_status = True



def run_exp_ham(num_qubits = 1, t  = 1, num_tests = 1):
    used_memory = virtual_memory().used
    
    for i in range(num_tests):
        
        if print_mem_status: print("Line ", lineno(), "\t Memory % used : ", virtual_memory().used - used_memory)
        used_memory = virtual_memory().used
        ham = h.random_hamiltonian(num_qubits)

#        if print_mem_status: print("Line ", lineno(), "\t Memory % used : ", virtual_memory().used - used_memory)
#        used_memory = virtual_memory().used

        expd = h.exp_ham(ham, t, enable_sparse_functionality=True)
#        expd = h.exp_ham(ham, t, enable_sparse_functionality=False)
#        expd = linalg.expm(-1j*ham*t)
#        if print_mem_status: print("Line ", lineno(), "\t Memory % used : ", virtual_memory().used - used_memory)
#        used_memory = virtual_memory().used

        del ham
        del expd
#        if print_mem_status: print("Line ", lineno(), "\t Memory % used : ", virtual_memory().used - used_memory)
 #       used_memory = virtual_memory().used







if __name__ == '__main__':
    orig = sys.stdout
    f = open('mem_out.txt', 'w')
    sys.stdout = f

    run_exp_ham(num_qubits = 7, num_tests = 10)
    
    sys.stdout = orig
    f.close()
