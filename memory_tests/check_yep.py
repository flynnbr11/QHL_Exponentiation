import yep
import hamiltonian_exponentiation as h

num_qubits = 2
ham = h.random_hamiltonian(num_qubits)
t=1

yep.start('yep_output.prof')
h.exp_ham(ham, t)
yep.stop()

