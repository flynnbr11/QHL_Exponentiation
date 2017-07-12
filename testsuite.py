import matrix_utils as mu
import numpy as np

#mu.test_run()

mat = np.identity(8, dtype=np.complex128)
emat = mu.expm_special(mat)

print emat

