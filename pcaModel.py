import numpy as np


def norm_from_optimal_pca(pca_matrix, k, heart_beat):
    uk = pca_matrix[:, :k]
    u_ut = np.matmul(uk, uk.T)
    distance = np.linalg.norm(heart_beat - (np.matmul(u_ut, heart_beat)))
    return distance


def get_pca_matrix(train_hb):
    hb_matrix = np.zeros(150)
    for nb in train_hb:
        hb_matrix = np.vstack((hb_matrix, nb))
    hb_matrix = hb_matrix[1:]
    hb_matrix = np.array(hb_matrix, dtype='int')
    hb_matrix = hb_matrix.T
    u, d, v_t = np.linalg.svd(np.matmul(hb_matrix, hb_matrix.T))
    return u


def get_distances(matrix, heart_beats):
    number_of_components = 45
    dist_coll = np.zeros(1)
    for hb in heart_beats:
        dist = norm_from_optimal_pca(matrix, number_of_components, hb)
        dist_coll = np.append(dist_coll, dist)
    dist_coll = dist_coll[1:]
    dist_coll = dist_coll.reshape(-1, 1)
    return dist_coll
