import numpy as np

def generalized_davidson_diagonalization(matrix, n_eigenvalues, search_dim_multiplier = 2,
                                         eigs_tol = 1e-10, eigvecs_tol = 1e-8, max_iter = 1000) :
    n_rows, n_cols = matrix.shape
    assert(n_rows == n_cols)

    iteration_vectors = np.eye(n_rows, n_eigenvalues) + 1
    guess_eigenvalues = np.zeros(n_eigenvalues)

    for iter in range(max_iter):

        old_eigenvalues = guess_eigenvalues[:n_eigenvalues]

        # perform QR decomposition to make sure the column vectors are orthonormal
        iteration_vectors, upper_triangular = np.linalg.qr(iteration_vectors)

        weight_matrix = np.dot(matrix, iteration_vectors)
        rayleigh_matrix = np.dot(iteration_vectors.T, weight_matrix)
        guess_eigenvalues, guess_eigenvectors = np.linalg.eig(rayleigh_matrix)

        sorted_indices = guess_eigenvalues.argsort()
        guess_eigenvalues = guess_eigenvalues[sorted_indices]
        guess_eigenvectors = guess_eigenvectors[:, sorted_indices]

        ritz_vectors = np.dot(iteration_vectors, guess_eigenvectors[:, :n_eigenvalues])

        eigenvalues_convergence_flag = np.all(np.abs(guess_eigenvalues[:n_eigenvalues] - old_eigenvalues) < eigs_tol)

        residuals = np.dot(ritz_vectors, np.diagflat(guess_eigenvalues[:n_eigenvalues])) - \
                    np.dot(weight_matrix, guess_eigenvectors[:, :n_eigenvalues])

        residual_convergence_flag = np.all(np.abs(residuals) < eigvecs_tol)
        if residual_convergence_flag and eigenvalues_convergence_flag:
            break

        new_directions = ritz_vectors

        for i in range(n_eigenvalues):
            new_directions[:, i] = np.dot(np.diagflat(1.0 / (guess_eigenvalues[i] - np.diagonal(matrix))),
                                          np.dot(guess_eigenvalues[i] * np.eye(n_rows, n_cols) - matrix,
                                                 ritz_vectors[:, i]))

        search_n_rows, search_n_cols = iteration_vectors.shape

        if search_n_cols <= n_eigenvalues * (search_dim_multiplier - 1):
            iteration_vectors = np.concatenate((iteration_vectors, residuals), axis=1)
        else:
            iteration_vectors = np.concatenate((ritz_vectors, residuals), axis=1)


    if iter == max_iter:
        raise Exception("Davidson diagonaliztion failed")

    return guess_eigenvalues[:n_eigenvalues]

def jacobi_davidson_diagonalization(transformer,
                                    diagonal,
                                    eigenvalue_index,
                                    start_search_dim,
                                    n_dim,
                                    residue_tol=1e-8,
                                    max_iter=1000):

    search_space = np.eye(n_dim, start_search_dim) + 0.01

    for iter in range(max_iter):

        # perform QR decomposition to make sure the column vectors are orthonormal
        orthonormal_subspace, upper_triangular = np.linalg.qr(search_space)

        M = orthonormal_subspace.shape[1]

        Ab_i = np.zeros((n_dim, M))

        for i in range(M):

            Ab_i[:, i] = transformer(orthonormal_subspace[:, i])

        interaction_matrix = np.dot(orthonormal_subspace.T, Ab_i)
        eigs, eigvecs = np.linalg.eig(interaction_matrix)

        sorted_indices = eigs.argsort()
        eig = eigs[sorted_indices[eigenvalue_index]]
        eigvec = eigvecs[:, sorted_indices[eigenvalue_index]]

        residue = np.dot(Ab_i, eigvec) - eig * np.dot(orthonormal_subspace, eigvec)

        if np.linalg.norm(residue) < residue_tol:
            print(iter)
            return eig, eigvec

        xi = np.dot(np.diagflat(1.0 / (eig - diagonal)), residue)

        np.eye(n_dim) - np.einsum('ij, kj -> jik', orthonormal_subspace, orthonormal_subspace)

        search_space = np.concatenate((orthonormal_subspace, np.array([xi]).T), axis=1)

    raise Exception("Davidson diagonaliztion failed")