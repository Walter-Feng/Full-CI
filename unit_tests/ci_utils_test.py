import src.ci_utils
import numpy as np


assert(src.ci_utils.compare_excitation({0, 1, 2}, {0, 1, 2}) == (set(), set()))
assert(src.ci_utils.compare_excitation({0, 1, 2}, {0, 1, 3}) == ({2}, {3}))
assert(src.ci_utils.compare_excitation({0, 2, 3}, {0, 1, 3}) == ({2}, {1}))
assert(src.ci_utils.compare_excitation({0, 2, 3, 4}, {0, 1, 5, 6}) == ({2,3,4}, {1, 5, 6}))

h1e = np.load("doc/h1e.npy")
h2e = np.load("doc/h2e.npy")

hamiltonian = src.ci_utils.ci_hamiltonian(h1e, h2e, 6)
eigvals = src.ci_utils.ci_direct_diagonalize(h1e, h2e, 6)
assert(np.abs(np.sort(eigvals)[0] + 7.8399080148963369) < 1e-13)

rand_vector = np.random.rand(eigvals.shape[0])

assert(np.all(np.abs(np.dot(hamiltonian, rand_vector) - src.ci_utils.ci_transform(rand_vector, h1e, h2e, 6))) < 1e-16)

ci_diagonal, ci_sparse_matrix = ci_utils.ci_hamiltonian_in_sparse_matrix(h1e, h2e, 6)

assert(
    np.abs(matrix_utils.jacobi_davidson_diagonalization(
        lambda vec: matrix_utils.sparse_matrix_transform(ci_sparse_matrix, vec),
        ci_diagonal,
        0,
        2,
        400,
        residue_tol=1e-5
)[0] + 7.8399080148963369) < 1e-10)

