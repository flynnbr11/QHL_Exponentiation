import matrix_utils as mu
import numpy as np

#mu.test_run()

mat = np.zeros([8, 8], dtype=np.complex128)
emat = mu.expm_special(mat)

print emat

