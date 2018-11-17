import numpy as np

from tqdm import tqdm


def compute_neighborhood_score(neighborhood2node, node2attribute, neighborhood_score_type):

    with np.errstate(invalid='ignore', divide='ignore'):
        A = neighborhood2node
        B = np.where(~np.isnan(node2attribute), node2attribute, 0)

        NA = A
        NB = np.where(~np.isnan(node2attribute), 1, 0)

        AB = np.dot(A, B)  # sum of attribute values in a neighborhood

        neighborhood_score = AB

        if neighborhood_score_type == 'z-score':
            N = np.dot(NA, NB)  # number of not-NaNs values in a neighborhood

            M = np.divide(AB, N)  # average attribute value in a neighborhood

            EXX = np.divide(np.dot(A, np.power(B, 2)), N)
            EEX = np.power(M, 2)

            std = np.sqrt(EXX - EEX)  # standard deviation of attribute values in a neighborhood

            neighborhood_score = np.divide(M, std)
            neighborhood_score[std == 0] = np.nan
            neighborhood_score[N < 3] = np.nan

    return neighborhood_score


def run_permutations(arg_tuple):

    # Seed the random number generator to a "random" number
    np.random.seed()

    neighborhood2node, node2attribute, neighborhood_score_type, num_permutations = arg_tuple

    N_in_neighborhood_in_group = compute_neighborhood_score(neighborhood2node, node2attribute, neighborhood_score_type)

    n2a = np.copy(node2attribute)
    indx_vals = np.nonzero(np.sum(~np.isnan(n2a), axis=1))[0]

    counts_neg = np.zeros(N_in_neighborhood_in_group.shape)
    counts_pos = np.zeros(N_in_neighborhood_in_group.shape)

    for _ in tqdm(np.arange(num_permutations)):
        # Permute only the rows that have values
        n2a[indx_vals, :] = n2a[np.random.permutation(indx_vals), :]

        N_in_neighborhood_in_group_perm = compute_neighborhood_score(neighborhood2node,
                                                                     n2a,
                                                                     neighborhood_score_type)

        with np.errstate(invalid='ignore', divide='ignore'):
            counts_neg = np.add(counts_neg, N_in_neighborhood_in_group_perm <= N_in_neighborhood_in_group)
            counts_pos = np.add(counts_pos, N_in_neighborhood_in_group_perm >= N_in_neighborhood_in_group)

    # print('Finished %d permutations.' % num_permutations)

    return counts_neg, counts_pos
