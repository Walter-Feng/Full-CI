import numpy as np
import src.matrix_utils
import itertools
import math

# Compare the excitation, i.e. the mismatching orbital indices
def compare_excitation(left_indices, right_indices):
    left_indices_set = set(left_indices)
    right_indices_set = set(right_indices)

    # elements in the left set that are not elements in the right one
    unique_from_left = left_indices_set - right_indices_set
    unique_from_right = right_indices_set - left_indices_set
    return (unique_from_left, unique_from_right)

# Calculate the phase factor, i.e. the sign of the determinant when the creation/annihilation operators are swapped to
# the tail of the list of operators
def phase_factor(excitation, left_indices, right_indices):
    indices_swap = 0
    left_excitation, right_excitation = excitation

    # check the left and right indices are in ascending order
    assert(sorted(left_indices) and sorted(right_indices))

    # check the excitations are in ascending order
    assert((sorted(left_excitation) or len(left_excitation) == 0)
           and (sorted(right_excitation) or len(right_excitation) == 0))

    # For a list of numbers in ascending order (which is checked in the assertions above), the number of swaps needed
    # to move a number to the head of the list is determined by its index in the list. e.g. in the following list
    # 1 4 5 6 8
    #       ^
    # 0 1 2 3 4  <---- index of the list in python
    # you need three 2-cycles to move 6 to the head of the list.
    # If you have two or more numbers, you need to remove the redundant swapping to keep the operators
    # corresponding to excited orbitals in ascending order. e.g.
    # 1 4 5 6 8
    #       ^ ^
    # 0 1 2 3 4  <---- index of the list in python
    # to move the two numbers we first add the indices of two numbers in the list to get the number of swaps for the
    # list
    # 8 6 1 4 5
    # and one step is redundant because we had 8 before 6, to correct this we subtract 1 from the number of swaps to
    # represent
    # 6 8 1 4 5
    # and this subtraction corresponds to the index of the excitation orbital in excitation list.
    for index, orbital_index in enumerate(left_excitation):
        indices_swap += left_indices.index(orbital_index) - index

    for index, orbital_index in enumerate(right_excitation):
        indices_swap += right_indices.index(orbital_index) - index

    return math.pow(-1, indices_swap)

def ci_hamiltonian_in_sparse_matrix(one_electron_integrals, two_electron_integrals, n_elecs, n_spin = 0) :

    n_rows, n_cols = one_electron_integrals.shape

    n_orbs = n_rows

    assert(n_rows == n_cols)
    assert(np.all(np.array(two_electron_integrals.shape) == n_orbs))

    n_alpha = (n_elecs + n_spin) // 2
    n_beta = (n_elecs - n_spin) // 2

    # This generates all possible combinations of the occupied orbitals, with indices of the orbitals in ascending order
    alpha_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    # the dimension of the hamiltonian matrix (dimension of the determinant basis)
    n_dim = len(alpha_combinations) * len(beta_combinations)

    non_trivial = []
    diagonal = []

    for i in range(n_dim):

        # The index of configuration is interpreted as follows:
        # Think of it as a two-digit number,
        #         2                    1
        #  index for alpha       index for beta
        # and when the number is added one the index for beta electrons is added first.
        # If the index for beta electrons exceeds the boundary (number of possible beta electron configurations),
        # the index becomes 0, and the index for alpha is added by 1.
        # This iterates over the configurations with each being unique.
        # The index for beta is modulo of the index i divided by the number of beta configurations,
        # and the index for alpha is the floor integer of the division.
        i_alpha_combination = alpha_combinations[i % len(beta_combinations)]
        i_beta_combination = beta_combinations[i // len(beta_combinations)]

        for j in range(i, n_dim):

            j_alpha_combination = alpha_combinations[j % len(beta_combinations)]
            j_beta_combination = beta_combinations[j // len(beta_combinations)]

            alpha_excitation = compare_excitation(i_alpha_combination, j_alpha_combination)
            beta_excitation = compare_excitation(i_beta_combination, j_beta_combination)

            n_alpha_excitation = len(alpha_excitation[0])
            n_beta_excitation = len(beta_excitation[0])

            # more than two electrons are excited - the matrix element is zero
            if n_alpha_excitation + n_beta_excitation > 2:
                continue

            # the phase factor for alpha and beta each
            alpha_phase = phase_factor(alpha_excitation, i_alpha_combination, j_alpha_combination)
            beta_phase = phase_factor(beta_excitation, i_beta_combination, j_beta_combination)

            # I think of the alignment of the creation operators as something like
            #  | beta |  | alpha |
            #   1  3  5   2  5  6  |0>
            total_phase_factor = alpha_phase * beta_phase

            # No excitation, i.e. the Slater determinant is the same on the two sides
            if n_alpha_excitation + n_beta_excitation == 0:

                one_electron_part = \
                    np.einsum("ii->", one_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination)]) \
                    + np.einsum("ii->", one_electron_integrals[np.ix_(i_beta_combination, i_beta_combination)])

                # <ij | v | ij>, or (ii | jj). Non trivial contribution from configurations having
                # the same spin for i and the same spin for j
                coulomb_part = \
                    np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                      i_beta_combination, i_beta_combination)]) \
                    + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                              i_alpha_combination, i_alpha_combination)]) \
                    + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_beta_combination, i_beta_combination,
                                                                              i_beta_combination, i_beta_combination)])

                # <ij | v | ji>, or (ij | ji).
                # i and j must have the same spin, and thus the mixed spin terms are omitted
                exchange_part = \
                    0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                            i_alpha_combination, i_alpha_combination)]) + \
                    0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_beta_combination, i_beta_combination,
                                                                            i_beta_combination, i_beta_combination)])

                element = one_electron_part + coulomb_part - exchange_part
                non_trivial.append({
                    "index": (i, j),
                    "element": element,
                    "phase_factor": 1
                })

                diagonal.append(element)

            if n_alpha_excitation + n_beta_excitation == 1:
                alpha_shared_orbitals = list(set(i_alpha_combination).intersection(set(j_alpha_combination)))
                beta_shared_orbitals = list(set(i_beta_combination).intersection(set(j_beta_combination)))

                concatenated = alpha_shared_orbitals + beta_shared_orbitals

                if n_alpha_excitation == 1:

                    index_a = list(alpha_excitation[0])[0]
                    index_b = list(alpha_excitation[1])[0]

                    # <a i | v | b i>, or (ab | ii).
                    coulomb_submatrix = two_electron_integrals[np.ix_([index_a], [index_b], concatenated, concatenated)]

                    # <a i | v | i b>, or (ai | ib).
                    # a and i must have same spin (a and b are already have the same spin)
                    exchange_submatrix = two_electron_integrals[
                        np.ix_([index_a], alpha_shared_orbitals, alpha_shared_orbitals, [index_b])]

                else:
                    index_a = list(beta_excitation[0])[0]
                    index_b = list(beta_excitation[1])[0]

                    coulomb_submatrix = two_electron_integrals[np.ix_([index_a], [index_b], concatenated, concatenated)]
                    exchange_submatrix = two_electron_integrals[
                        np.ix_([index_a], beta_shared_orbitals, beta_shared_orbitals, [index_b])]

                element = \
                    one_electron_integrals[index_a, index_b] + \
                          np.einsum("ijkk->", coulomb_submatrix) - np.einsum("ikkj->", exchange_submatrix)

                non_trivial.append({
                    "index": (i, j),
                    "element": total_phase_factor * element,
                    "phase_factor": total_phase_factor
                })

            if n_alpha_excitation == 2:
                left_excitation, right_excitation = map(list, alpha_excitation)

                # <ab | v | xy> - <ab | v | yx>, or (ax | by) - (ay | bx)
                element = two_electron_integrals[left_excitation[0], right_excitation[0],
                                                 left_excitation[1], right_excitation[1]] \
                          - two_electron_integrals[left_excitation[0], right_excitation[1],
                                                   left_excitation[1], right_excitation[0]]

                non_trivial.append({
                    "index": (i, j),
                    "element": total_phase_factor * element,
                    "phase_factor": total_phase_factor
                })

            if n_beta_excitation == 2:
                left_excitation, right_excitation = map(list, beta_excitation)

                # <ab | v | xy> - <ab | v | yx>, or (ax | by) - (ay | bx)
                element = two_electron_integrals[left_excitation[0], right_excitation[0],
                                                 left_excitation[1], right_excitation[1]] \
                          - two_electron_integrals[left_excitation[0], right_excitation[1],
                                                   left_excitation[1], right_excitation[0]]

                non_trivial.append({
                    "index": (i, j),
                    "element": total_phase_factor * element,
                    "phase_factor": total_phase_factor
                })

            if n_alpha_excitation == 1 and n_beta_excitation == 1:
                a = list(alpha_excitation[0])[0]
                b = list(beta_excitation[0])[0]
                x = list(alpha_excitation[1])[0]
                y = list(beta_excitation[1])[0]

                element = two_electron_integrals[a, x, b, y]

                non_trivial.append({
                    "index": (i, j),
                    "element": total_phase_factor * element,
                    "phase_factor": total_phase_factor
                })

    return np.array(diagonal), non_trivial

###########################################################################################
#
#
#         All the functions below were for debugging purposes
#
#
###########################################################################################
def ci_hamiltonian(one_electron_integrals, two_electron_integrals, n_elecs, n_spin = 0):

    n_rows, n_cols = one_electron_integrals.shape

    n_orbs = n_rows

    assert(n_rows == n_cols)
    assert(np.all(np.array(two_electron_integrals.shape) == n_orbs))

    n_alpha = (n_elecs + n_spin) // 2
    n_beta = (n_elecs - n_spin) // 2

    diagonal, non_trivial = \
        ci_hamiltonian_in_sparse_matrix(one_electron_integrals, two_electron_integrals, n_elecs, n_spin)

    n_dim = diagonal.shape[0]

    hamiltonian = np.zeros((n_dim, n_dim))

    for i in non_trivial:
        row, col = i["index"]
        hamiltonian[row, col] = i["element"]
        if row != col:
            hamiltonian[col, row] = i["element"]

    return hamiltonian

def ci_hamiltonian_diagonal(one_electron_integrals, two_electron_integrals, n_elecs, n_spin = 0) :

    n_rows, n_cols = one_electron_integrals.shape

    n_orbs = n_rows

    assert(n_rows == n_cols)
    assert(np.all(np.array(two_electron_integrals.shape) == n_orbs))

    n_alpha = (n_elecs + n_spin) // 2
    n_beta = (n_elecs - n_spin) // 2

    # This generates all possible combinations of the occupied orbitals, with indices of the orbitals in ascending order
    alpha_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    # the dimension of the hamiltonian matrix (dimension of the determinant basis)
    n_dim = len(alpha_combinations) * len(beta_combinations)

    diagonal = np.zeros(n_dim)

    for i in range(n_dim):

        # The index of configuration is interpreted as follows:
        # Think of it as a two-digit number,
        #         2                    1
        #  index for alpha       index for beta
        # and when the number is added one the index for beta electrons is added first.
        # If the index for beta electrons exceeds the boundary (number of possible beta electron configurations),
        # the index becomes 0, and the index for alpha is added by 1.
        # This iterates over the configurations with each being unique.
        # The index for beta is modulo of the index i divided by the number of beta configurations,
        # and the index for alpha is the floor integer of the division.
        i_alpha_combination = alpha_combinations[i % len(beta_combinations)]
        i_beta_combination = beta_combinations[i // len(beta_combinations)]

        one_electron_part = \
            np.einsum("ii->", one_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination)]) \
            + np.einsum("ii->", one_electron_integrals[np.ix_(i_beta_combination, i_beta_combination)])

        # <ij | v | ij>, or (ii | jj). Non trivial contribution from  configurations having
        # the same spin for i and the same spin for j
        coulomb_part = \
            np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                              i_beta_combination, i_beta_combination)]) \
            + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                      i_alpha_combination, i_alpha_combination)]) \
            + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_beta_combination, i_beta_combination,
                                                                      i_beta_combination, i_beta_combination)])

        # <ij | v | ji>, or (ij | ji).
        # i and j must have the same spin, and thus the mixed spin terms are omitted
        exchange_part = \
            0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                    i_alpha_combination, i_alpha_combination)]) + \
            0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_beta_combination, i_beta_combination,
                                                                    i_beta_combination, i_beta_combination)])

        diagonal[i] = one_electron_part + coulomb_part - exchange_part

    return diagonal


def ci_direct_diagonalize(one_electron_integrals, two_electron_integrals, n_elecs, n_spin = 0) :
    return np.linalg.eigvals(ci_hamiltonian(one_electron_integrals, two_electron_integrals, n_elecs, n_spin=n_spin))


def ci_transform(config_vector, one_electron_integrals, two_electron_integrals, n_elecs, n_spin = 0) :

    n_rows, n_cols = one_electron_integrals.shape

    n_orbs = n_rows

    assert(n_rows == n_cols)
    assert(np.all(np.array(two_electron_integrals.shape) == n_orbs))


    n_alpha = (n_elecs + n_spin) // 2
    n_beta = (n_elecs - n_spin) // 2

    # This generates all possible combinations of the occupied orbitals, with indices of the orbitals in ascending order
    alpha_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    # the dimension of the hamiltonian matrix (dimension of the determinant basis)
    n_dim = len(alpha_combinations) * len(beta_combinations)

    assert (config_vector.shape[0] == n_dim)

    transformed_vector = np.zeros(n_dim)

    for i in range(n_dim):

        # The index of configuration is interpreted as follows:
        # Think of it as a two-digit number,
        #         2                    1
        #  index for alpha       index for beta
        # and when the number is added one the index for beta electrons is added first.
        # If the index for beta electrons exceeds the boundary (number of possible beta electron configurations),
        # the index becomes 0, and the index for alpha is added by 1.
        # This iterates over the configurations with each being unique.
        # The index for beta is modulo of the index i divided by the number of beta configurations,
        # and the index for alpha is the floor integer of the division.
        i_alpha_combination = alpha_combinations[i % len(beta_combinations)]
        i_beta_combination = beta_combinations[i // len(beta_combinations)]

        for j in range(i, n_dim):

            j_alpha_combination = alpha_combinations[j % len(beta_combinations)]
            j_beta_combination = beta_combinations[j // len(beta_combinations)]

            alpha_excitation = compare_excitation(i_alpha_combination, j_alpha_combination)
            beta_excitation = compare_excitation(i_beta_combination, j_beta_combination)

            n_alpha_excitation = len(alpha_excitation[0])
            n_beta_excitation = len(beta_excitation[0])

            # more than two electrons are excited - the matrix element is zero
            if n_alpha_excitation + n_beta_excitation > 2:
                continue

            # the phase factor for alpha and beta each
            alpha_phase = phase_factor(alpha_excitation, i_alpha_combination, j_alpha_combination)
            beta_phase = phase_factor(beta_excitation, i_beta_combination, j_beta_combination)

            # I think of the alignment of the creation operators as something like
            #  | beta |  | alpha |
            #   1  3  5   2  5  6  |0>
            total_phase_factor = alpha_phase * beta_phase

            # No excitation, i.e. the Slater determinant is the same on the two sides
            if n_alpha_excitation + n_beta_excitation == 0:

                one_electron_part = \
                    np.einsum("ii->", one_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination)]) \
                    + np.einsum("ii->", one_electron_integrals[np.ix_(i_beta_combination, i_beta_combination)])

                # <ij | v | ij>, or (ii | jj). Non trivial contribution from  configurations having
                # the same spin for i and the same spin for j
                coulomb_part = \
                    np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                      i_beta_combination, i_beta_combination)]) \
                    + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                              i_alpha_combination, i_alpha_combination)]) \
                    + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_beta_combination, i_beta_combination,
                                                                              i_beta_combination, i_beta_combination)])

                # <ij | v | ji>, or (ij | ji).
                # i and j must have the same spin, and thus the mixed spin terms are omitted
                exchange_part = \
                    0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                            i_alpha_combination, i_alpha_combination)]) + \
                    0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_beta_combination, i_beta_combination,
                                                                            i_beta_combination, i_beta_combination)])

                transformed_vector[i] += (one_electron_part + coulomb_part - exchange_part) * config_vector[j]

            if n_alpha_excitation + n_beta_excitation == 1:
                alpha_shared_orbitals = list(set(i_alpha_combination).intersection(set(j_alpha_combination)))
                beta_shared_orbitals = list(set(i_beta_combination).intersection(set(j_beta_combination)))

                concatenated = alpha_shared_orbitals + beta_shared_orbitals

                if n_alpha_excitation == 1:

                    index_a = list(alpha_excitation[0])[0]
                    index_b = list(alpha_excitation[1])[0]

                    # <a i | v | b i>, or (ab | ii).
                    coulomb_submatrix = two_electron_integrals[np.ix_([index_a], [index_b], concatenated, concatenated)]

                    # <a i | v | i b>, or (ai | ib).
                    # a and i must have same spin (a and b are already have the same spin)
                    exchange_submatrix = two_electron_integrals[
                        np.ix_([index_a], alpha_shared_orbitals, alpha_shared_orbitals, [index_b])]

                else:
                    index_a = list(beta_excitation[0])[0]
                    index_b = list(beta_excitation[1])[0]

                    coulomb_submatrix = two_electron_integrals[np.ix_([index_a], [index_b], concatenated, concatenated)]
                    exchange_submatrix = two_electron_integrals[
                        np.ix_([index_a], beta_shared_orbitals, beta_shared_orbitals, [index_b])]

                element = \
                    one_electron_integrals[index_a, index_b] + \
                          np.einsum("ijkk->", coulomb_submatrix) - np.einsum("ikkj->", exchange_submatrix)
                transformed_vector[i] += total_phase_factor * element * config_vector[j]
                transformed_vector[j] += total_phase_factor * element * config_vector[i]

            if n_alpha_excitation == 2:
                left_excitation, right_excitation = map(list, alpha_excitation)

                # <ab | v | xy> - <ab | v | yx>, or (ax | by) - (ay | bx)
                element = two_electron_integrals[left_excitation[0], right_excitation[0],
                                                 left_excitation[1], right_excitation[1]] \
                          - two_electron_integrals[left_excitation[0], right_excitation[1],
                                                   left_excitation[1], right_excitation[0]]

                transformed_vector[i] += total_phase_factor * element * config_vector[j]
                transformed_vector[j] += total_phase_factor * element * config_vector[i]

            if n_beta_excitation == 2:
                left_excitation, right_excitation = map(list, beta_excitation)

                # <ab | v | xy> - <ab | v | yx>, or (ax | by) - (ay | bx)
                element = two_electron_integrals[left_excitation[0], right_excitation[0],
                                                 left_excitation[1], right_excitation[1]] \
                          - two_electron_integrals[left_excitation[0], right_excitation[1],
                                                   left_excitation[1], right_excitation[0]]

                transformed_vector[i] += total_phase_factor * element * config_vector[j]
                transformed_vector[j] += total_phase_factor * element * config_vector[i]

            if n_alpha_excitation == 1 and n_beta_excitation == 1:
                a = list(alpha_excitation[0])[0]
                b = list(beta_excitation[0])[0]
                x = list(alpha_excitation[1])[0]
                y = list(beta_excitation[1])[0]

                element = two_electron_integrals[a, x, b, y]

                transformed_vector[i] += total_phase_factor * element * config_vector[j]
                transformed_vector[j] += total_phase_factor * element * config_vector[i]

    return transformed_vector

