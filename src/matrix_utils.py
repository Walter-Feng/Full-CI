import numpy as np

def davidson_diagonalization(matrix, n_eigenvalues, search_dim_multiplier = 2, eigs_tol = 1e-6, eigvecs_tol = 1e-5, max_iter = 1000) :
    n_rows, n_cols = matrix.shape
    assert(n_rows == n_cols)

    iteration_vectors = np.eye(n_rows, n_eigenvalues)
    guess_eigenvalues = np.zeros(n_eigenvalues)

    for iter in range(max_iter):
        old_eigenvalues = guess_eigenvalues[:n_eigenvalues]

        weight_matrix = np.dot(matrix, iteration_vectors)
        rayleigh_matrix = np.dot(iteration_vectors.T, weight_matrix)
        guess_eigenvalues, guess_eigenvectors = np.linalg.eig(rayleigh_matrix)

        sorted_indices = guess_eigenvalues.argsort()
        guess_eigenvalues = guess_eigenvalues[sorted_indices]
        guess_eigenvectors = guess_eigenvectors[:, sorted_indices]

        ritz_vectors = np.dot(iteration_vectors, guess_eigenvectors[:, :n_eigenvalues])

        eigenvalues_convergence_flag = np.all(np.abs(guess_eigenvalues[:n_eigenvalues] - old_eigenvalues) < eigs_tol)

        residuals = np.dot(ritz_vectors, np.diagflat(guess_eigenvalues[:n_eigenvalues])) - np.dot(weight_matrix, guess_eigenvectors[:, :n_eigenvalues])

        residual_convergence_flag = np.all(np.abs(residuals) < eigvecs_tol)
        if residual_convergence_flag and eigenvalues_convergence_flag:
            break

        new_directions = ritz_vectors

        for i in range(n_eigenvalues):
            new_directions[:, i] = np.dot(np.diagflat(1.0 / (guess_eigenvalues[i] - np.diagonal(matrix))),
                                          np.dot(guess_eigenvalues[i] * np.eye(n_rows, n_cols) - matrix, ritz_vectors[:, i]))

        search_n_rows, search_n_cols = iteration_vectors.shape

        if search_n_cols <= n_eigenvalues * (search_dim_multiplier - 1):
            iteration_vectors = np.concatenate((iteration_vectors, residuals), axis=1)
        else :
            iteration_vectors = np.concatenate((ritz_vectors, residuals), axis=1)

        # perform QR decomposition to make sure the column vectors are orthonormal
        iteration_vectors, upper_triangular = np.linalg.qr(iteration_vectors)

    if iter == max_iter:
        raise Exception("Davidson diagonaliztion failed")

    return guess_eigenvalues[:n_eigenvalues]