from __future__ import print_function # so print doesn't show brackets
import hamiltonian_exponentiation as h 
from psutil import virtual_memory

import sys, os        
import inspect
from memory_profiler import profile
from scipy import linalg
import random



def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno


print_mem_status = True


@profile
def run_exp_ham(num_qubits = 1, t  = 1, num_tests = 1, use_sparse=True):
    if use_sparse==True:
        print("\n\nSPARSE")
    elif use_sparse==False:
        print("\n\nNONSPARSE")
    else:
        print("use_sparse must be True or False")

    for i in range(num_tests):
        print("Num qubits : ", num_qubits)
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)
        ham = h.random_hamiltonian(num_qubits)
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)
        expd = h.exp_ham(ham, t, enable_sparse_functionality=use_sparse)
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)
        del ham
        del expd
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)


@profile
def run_linalg(num_qubits = 1, t  = 1, num_tests = 1):
    print("\n\nLINALG")
    for i in range(num_tests):
        print("Num qubits : ", num_qubits)
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)
        ham = h.random_hamiltonian(num_qubits)
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)
        expd = linalg.expm(-1j*ham*t)
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)
        del ham
        del expd
        print("Line ", lineno(), "\t Memory % used : ", virtual_memory().free)



if __name__ == '__main__':
    num_qubits = 10
    time=random.random()
    num_tests = 10
    for sp in [True, False]:
        run_exp_ham(num_qubits = num_qubits, t=time, num_tests = num_tests, use_sparse=sp)

    run_linalg(num_qubits = num_qubits, t=time, num_tests = 2)




