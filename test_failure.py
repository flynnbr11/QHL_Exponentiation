import hamiltonian_exponentiation as h
import numpy as np
from scipy import linalg
import time as time



def diff_mtx(ham, t, prec):
	begin_exp = time.time()
	expd = h.exp_minus_i_h_t(ham, t, prec)
	end_exp = time.time()
	
	begin_lin = time.time()
	lin = linalg.expm(-1j*ham*t)
	end_lin = time.time()
	
	
	print("Times: \t Exp: ", end_exp-begin_exp, "; \t linalg: ", end_lin-begin_lin)
	return (expd - lin)


sigmax = np.array([[0+0j, 1+0j], [1+0j, 0+0j]])
sigmay = np.array([[0+0j, 0-1j], [0+1j, 0+0j]])
sigmaz = np.array([[1+0j, 0+0j], [0+0j, -1+0j]])

a = 110.9
ham = a*sigmax
size=4
ham = h.random_hamiltonian(size)
ham = a * ham
t = 90.0
prec = 1e-14

sig_x = diff_mtx(ham, t, prec)

#print("Difference :")
#print(sig_x)


#lin = linalg.expm(-1.j*ham*t)
#ex = h.exp_minus_i_h_t(ham, t, prec)
"""
print("Input:")
print(ham)

print("lin:")
print(lin)

print("exp:")
print(ex)

print("Max value in hamiltonian = ", np.max(ham))
"""

#print("Inf reached: ", inf_reached, " \t Max difference = ", np.max(sig_x))
print("precision = ", prec)	
print("Max difference = ", np.max(sig_x))
	

