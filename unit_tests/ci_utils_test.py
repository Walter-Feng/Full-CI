import src.ci_utils
import numpy as np
import itertools

orbital_combinations = [list(x) for x in itertools.combinations(range(8), 5)]

assert([orbital_combinations[i] for i in [src.ci_utils.address_array(j, 5, 8) for j in orbital_combinations]]
                == orbital_combinations)
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

ci_diagonal, ci_sparse_matrix = src.ci_utils.ci_hamiltonian_in_sparse_matrix(h1e, h2e, 6)

assert(
    np.abs(src.matrix_utils.davidson_diagonalization(
        lambda vec: src.matrix_utils.sparse_matrix_transform(ci_sparse_matrix, vec),
        ci_diagonal,
        0,
        2,
        400,
        residue_tol=1e-5
)[0] + 7.8399080148963369) < 1e-10)


assert(
    np.abs(src.matrix_utils.davidson_diagonalization(
        src.ci_utils.knowles_handy_full_ci_transformer(h1e, h2e, 6),
        ci_diagonal,
        0,
        2,
        400,
        residue_tol=1e-5
)[0] + 7.8399080148963369) < 1e-10)
