import numpy as np
import src.matrix_utils
import itertools

def compare_excitation(left_indices, right_indices):
    unique_from_left = left_indices - right_indices
    unique_from_right = right_indices - left_indices
    return (unique_from_left, unique_from_right)

def diagonalize_ci(one_electron_integrals, two_electron_integrals, n_elecs, n_spin = 0) :

    n_rows, n_cols = one_electron_integrals

    n_orbs = n_rows

    assert(n_rows == n_cols)
    assert(np.all(np.array(two_electron_integrals.shape)) == n_orbs)

    n_alpha = (n_elecs + n_spin) / 2
    n_beta = (n_elecs - n_spin) / 2

    alpha_combinations = [set(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [set(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    n_dim = len(alpha_combinations) * len(beta_combinations)

    hamiltonian = np.zeros(n_dim, n_dim)

    for i in range(n_dim):
        for j in range(i, n_dim):
            i_alpha_combination = alpha_combinations[i % len(beta_combinations)]
            i_beta_combination = beta_combinations[i % len(beta_combinations)]
            j_alpha_combination = alpha_combinations[j % len(beta_combinations)]
            j_beta_combination = beta_combinations[j % len(beta_combinations)]

            alpha_excitation = compare_excitation(i_alpha_combination, j_alpha_combination)
            beta_excitation = compare_excitation(i_beta_combination, j_beta_combination)

            n_alpha_excitation = len(alpha_excitation[0])
            n_beta_excitation = len(beta_excitation[0])

            if n_alpha_excitation + n_beta_excitation > 2:
                continue

            if n_alpha_excitation + n_beta_excitation == 0:
                hamiltonian[i, j] += np.trace(one_electron_integrals)
                for k in range(n_dim):
                    for l in range(n_dim):
                        hamiltonian[i, j] += 0.5 * (two_electron_integrals[i, j, i, j] - two_electron_integrals[i, j, j, i])

            if n_alpha_excitation + n_beta_excitation == 1:
                if n_alpha_excitation == 1:
                    index_a = list(alpha_excitation[0])[0]
                    index_b = list(alpha_excitation[1])[0]
                    element = one_electron_integrals[index_a, index_b] + sum([two_electron_integrals[index_a, i, index_b, i] - two_electron_integrals[index_a, i, i, index_b] for i in range(n_orbs)])
                    hamiltonian[i, j] = element
                    hamiltonian[j, i] = element

                else: #todo starting here
                    hamiltonian[i, j] = one_electron_integrals[list(alpha_excitation[0]), list(alpha_excitation[1])]
                    hamiltonian[j, i] = one_electron_integrals[list(alpha_excitation[0]), list(alpha_excitation[1])]

