from src import matrix_utils
import numpy as np

a = np.array([
    [1, 0.1, 0, 0, 0],
    [0.1, -2, 0.3, 0, 0],
    [0, 0.3, 2, 0.1, 0],
    [0, 0, 0.1, 3, -0.1],
    [0, 0, 0, -0.1, 10]
])

assert(np.all(np.abs(np.linalg.eigvals(a)[:3] - matrix_utils.davidson_diagonalization(a, 3)) < 1e-10))
