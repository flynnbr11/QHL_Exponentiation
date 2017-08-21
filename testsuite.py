import hamiltoniain_exponetiation as h
import numpy as np
import time
import os as os

import sys
sys.path.append(os.path.join("test_matrices"))

num_qubits=1
ham = h.random_hamiltonian(num_qubits)

t_start = time.time()
h.expn_hamilt(mat)
t_end = time.time()

print("Finished exponentiation of ", num_qubit, "-qubit matrix in ", t_end-t_start, " seconds.")
