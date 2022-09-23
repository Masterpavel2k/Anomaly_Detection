import numpy as np

from distance_model_evaluation import model_evaluation, new_model_evaluation
from pca import matrix_construction, norm_from_optimal_pca, pca_matrix_construction


def get_distances(matrix, heart_beats):
    dist_coll = np.zeros(1)
    for hb in heart_beats:
        dist = norm_from_optimal_pca(matrix, 45, hb)
        dist_coll = np.append(dist_coll, dist)
    dist_coll = dist_coll[1:]
    dist_coll = dist_coll.reshape(-1, 1)
    return dist_coll


if __name__ == '__main__':
    # construction of the heart beats matrix, parameter is the number of heart beats
    s_matrix = matrix_construction(10000)
    # construction of pca matrix from heart beats matrix
    pca_matrix = pca_matrix_construction(s_matrix)

    # model_evaluation(pca_matrix, get_distances)
    new_model_evaluation(pca_matrix, get_distances)
