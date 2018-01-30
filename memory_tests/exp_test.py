import hamiltonian_exponentiation as h

sparse = True
ham = h.random_hamiltonian(8)
t=1.4

#h.exp_ham(ham, t, enable_sparse_functionality = sparse)

h.test_exp_ham_function(max_qubit=7, test_sparse_speedup=True, num_tests = 1, threshold=1e-15)
