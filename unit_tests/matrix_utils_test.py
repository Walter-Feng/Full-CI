from src import matrix_utils
import numpy as np
import math
from src import ci_utils

a = np.array([
    [1, 0.1, 0, 0, 0],
    [0.1, -2, 0.3, 0, 0],
    [0, 0.3, 2, 0.1, 0],
    [0, 0, 0.1, 3, -0.1],
    [0, 0, 0, -0.1, 10]
])

assert(np.all(np.abs(np.linalg.eigvals(a)[:3] -
                     matrix_utils.generalized_davidson_diagonalization(a, 3)) < 1e-10))

h1e = np.load("doc/h1e.npy")
h2e = np.load("doc/h2e.npy")

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


