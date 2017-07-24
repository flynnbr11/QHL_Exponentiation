import matrix_utils as mu
import numpy as np
import time

#mu.test_run()

mat_size = 4
mat = np.identity(mat_size, dtype=np.complex128)

t_start = time.time()
emat = mu.expm_special(mat)
t_end = time.time()

print emat

print 'finished {}x{} in {} seconds.'.format(mat_size, mat_size, t_end - t_start)
