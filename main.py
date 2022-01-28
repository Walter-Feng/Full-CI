import src.ci_utils
import src.matrix_utils

def full_ci(one_electron_integrals, two_electron_integrals, n_elecs, start_search_dim, n_spin = 0, residue_tol=1e-5) :
    ci_diagonal, ci_sparse_matrix = src.ci_utils.ci_hamiltonian_in_sparse_matrix(one_electron_integrals,
                                                                             two_electron_integrals,
                                                                             n_elecs, n_spin)

    n_dim = ci_diagonal.shape[0]

    return src.matrix_utils.davidson_diagonalization(
        lambda vec: src.matrix_utils.sparse_matrix_transform(ci_sparse_matrix, vec),
        ci_diagonal,
        0,
        start_search_dim,
        n_dim,
        residue_tol
    )

def knowles_handy_full_ci(one_electron_integrals, two_electron_integrals,
                          n_elecs, start_search_dim, n_spin = 0, residue_tol=1e-5) :
