import numpy as np
import src.matrix_utils
import itertools
import math

def compare_excitation(left_indices, right_indices):
    left_indices_set = set(left_indices)
    right_indices_set = set(right_indices)

    unique_from_left = left_indices_set - right_indices_set
    unique_from_right = right_indices_set - left_indices_set
    return (unique_from_left, unique_from_right)

def phase_factor(excitation, left_indices, right_indices):
    indices_swap = 0
    left_excitation, right_excitation = excitation

    assert(sorted(left_indices) and sorted(right_indices))
    assert((sorted(left_excitation) or len(left_excitation) == 0) and (sorted(right_excitation) or len(right_excitation) == 0))

    for index, orbital_index in enumerate(left_excitation):
        indices_swap += orbital_index + index

    return math.pow(-1, indices_swap)

def diagonalize_ci(one_electron_integrals, two_electron_integrals, n_elecs, n_spin = 0) :

    n_rows, n_cols = one_electron_integrals.shape

    n_orbs = n_rows

    assert(n_rows == n_cols)
    assert(np.all(np.array(two_electron_integrals.shape) == n_orbs))

    n_alpha = (n_elecs + n_spin) // 2
    n_beta = (n_elecs - n_spin) // 2

    alpha_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    n_dim = len(alpha_combinations) * len(beta_combinations)

    hamiltonian = np.zeros((n_dim, n_dim))

    for i in range(n_dim):

        i_alpha_combination = alpha_combinations[i % len(beta_combinations)]
        i_beta_combination = beta_combinations[i // len(beta_combinations)]

        # for j in range(i, n_dim):
        for j in range(i, n_dim):

            j_alpha_combination = alpha_combinations[j % len(beta_combinations)]
            j_beta_combination = beta_combinations[j // len(beta_combinations)]

            alpha_excitation = compare_excitation(i_alpha_combination, j_alpha_combination)
            beta_excitation = compare_excitation(i_beta_combination, j_beta_combination)

            n_alpha_excitation = len(alpha_excitation[0])
            n_beta_excitation = len(beta_excitation[0])

            if n_alpha_excitation + n_beta_excitation > 2:
                continue

            alpha_phase = phase_factor(alpha_excitation, i_alpha_combination, j_alpha_combination)
            beta_phase = phase_factor(beta_excitation, i_beta_combination, j_beta_combination)

            total_phase_factor = alpha_phase * beta_phase

            if n_alpha_excitation + n_beta_excitation == 0:
                concatenated = list(i_alpha_combination) + list(i_beta_combination)
                one_electron_integral_index = np.ix_(concatenated, concatenated)
                two_electron_integrals_index = np.ix_(concatenated, concatenated, concatenated, concatenated)
                hamiltonian[i, j] = \
                    np.einsum("ii->", one_electron_integrals[one_electron_integral_index]) + \
                    0.5 * np.einsum("iijj->", two_electron_integrals[two_electron_integrals_index]) - \
                    0.5 * np.einsum("ijji->", two_electron_integrals[two_electron_integrals_index])

            if n_alpha_excitation + n_beta_excitation == 1:
                if n_alpha_excitation == 1:
                    index_a = list(alpha_excitation[0])[0]
                    index_b = list(alpha_excitation[1])[0]

                else:
                    index_a = list(beta_excitation[0])[0]
                    index_b = list(beta_excitation[1])[0]

                concatenated = \
                    list(set(i_alpha_combination).intersection(set(j_alpha_combination))) + \
                    list(set(i_beta_combination).intersection(set(j_beta_combination)))

                coulomb_submatrix = two_electron_integrals[np.ix_([index_a], [index_b], concatenated, concatenated)]
                exchange_submatrix = two_electron_integrals[np.ix_([index_a], concatenated, concatenated, [index_b])]
                element = \
                    one_electron_integrals[index_a, index_b] + np.einsum("ijkk->", coulomb_submatrix) - \
                    np.einsum("ikkj->", exchange_submatrix)
                hamiltonian[i, j] = total_phase_factor * element
                hamiltonian[j, i] = total_phase_factor * element

            if n_alpha_excitation + n_beta_excitation == 2:

                indices_a = np.sort(np.array(list(alpha_excitation[0]) + list(beta_excitation[0])))
                indices_b = np.sort(np.array(list(alpha_excitation[1]) + list(beta_excitation[1])))

                element = two_electron_integrals[indices_a[0], indices_b[0], indices_a[1], indices_b[1]] - \
                          two_electron_integrals[indices_a[0], indices_b[1], indices_a[1], indices_b[0]]

                hamiltonian[i, j] = total_phase_factor * element
                hamiltonian[j, i] = total_phase_factor * element

    # print(np.sort(np.linalg.eigvals(hamiltonian)))\
    print(hamiltonian[:5, :5])
    return np.sort(np.linalg.eigvals(hamiltonian)[:5])
    # return src.matrix_utils.davidson_diagonalization(hamiltonian, 2, search_dim_multiplier = 6)

def addressing_array(elec_index, orb_index, n_elec, n_orb):
    return sum([math.comb(m, n_elec - elec_index) - math.comb(m-1, n_elec - elec_index - 1)
                for m in range(n_orb - orb_index + 1, n_orb - elec_index + 1)])

