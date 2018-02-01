from __future__ import print_function # so print doesn't show brackets
import hamiltonian_exponentiation as h 
from psutil import virtual_memory

import sys, os        
import inspect
from memory_profiler import profile
from scipy import linalg
import sys
import random

def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno




print_mem_status = True
save_to_file = False


def run_exp_ham(num_qubits = 1, t  = 1, num_tests = 1, 
    test_custom_sparse=False, 
    test_custom=False, 
    test_linalg=False        
    ):
    used_memory = virtual_memory().used
    
    for i in range(num_tests):
        
        if print_mem_status and i>0: print("i=",i, "\t System memory used loop : ", virtual_memory().used)
        
        ham = h.random_hamiltonian(num_qubits)
        t = random.random()
        
        if test_custom_sparse:
            expd = h.exp_ham(ham, t, enable_sparse_functionality=True)
        elif test_custom:
            expd = h.exp_ham(ham, t, enable_sparse_functionality=False)
        elif test_linalg:
            expd = linalg.expm(-1j*ham*t)
        else:
            print("Must choose one method to test")
        
        del ham
        del expd



if __name__ == '__main__':
    
    if save_to_file:
        orig = sys.stdout
        f = open('mem_out.txt', 'w')
        sys.stdout = f

    run_exp_ham(num_qubits = 7, num_tests = 10, test_linalg=True)
    

    if save_to_file:
        sys.stdout = orig
        f.close()
