import src.ci_utils
import numpy as np


assert(src.ci_utils.compare_excitation({0, 1, 2}, {0, 1, 2}) == (set(), set()))
assert(src.ci_utils.compare_excitation({0, 1, 2}, {0, 1, 3}) == ({2}, {3}))
assert(src.ci_utils.compare_excitation({0, 2, 3}, {0, 1, 3}) == ({2}, {1}))
assert(src.ci_utils.compare_excitation({0, 2, 3, 4}, {0, 1, 5, 6}) == ({2,3,4}, {1, 5, 6}))

h1e = np.load("doc/h1e.npy")
h2e = np.load("doc/h2e.npy")
print(src.ci_utils.diagonalize_ci(h1e, h2e, 6))

