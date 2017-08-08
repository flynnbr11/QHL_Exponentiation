import matrix_utils as mu
import numpy as np
import time
import os as os

import sys
sys.path.append(os.path.join("test_matrices"))
import random_matrix as rm


#mu.test_run()



mat_size = 1
mat = np.identity(2**mat_size, dtype=np.complex128)
mat = rm.get_matrix(mat_size)
print 'Input Matrix : '
print mat
print ' --- ----'

t_start = time.time()
emat = mu.expn_hamilt(mat)
t_end = time.time()



print emat

print 'finished {}x{} in {} seconds.'.format(mat_size, mat_size, t_end - t_start)
