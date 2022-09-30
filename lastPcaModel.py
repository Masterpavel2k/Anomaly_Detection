import numpy as np
from pcaModel import get_pca_matrix


def first_pca_component(train_hb):
    pca = get_pca_matrix(train_hb)
    last_pca_comp = pca[:, 1].T
    return last_pca_comp


def last_pca_component(train_hb):
    pca = get_pca_matrix(train_hb)
    last_pca_comp = pca[:, -1].T
    return last_pca_comp


def get_distances(pca_comp, heart_beats):
    dist_coll = np.zeros(1)
    for hb in heart_beats:
        dist = np.matmul(pca_comp, hb)
        dist_coll = np.append(dist_coll, abs(dist))
    dist_coll = dist_coll[1:]
    dist_coll = dist_coll.reshape(-1, 1)
    return dist_coll
