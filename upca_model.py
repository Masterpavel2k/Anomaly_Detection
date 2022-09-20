import numpy as np

from distance_model_evaluation import model_evaluation
from upca import first_pca_component, last_pca_component


def get_distances(pca_comp, heart_beats):
    dist_coll = np.zeros(1)
    for hb in heart_beats:
        dist = np.matmul(pca_comp, hb)
        dist_coll = np.append(dist_coll, abs(dist))
    dist_coll = dist_coll[1:]
    dist_coll = dist_coll.reshape(-1, 1)
    return dist_coll


if __name__ == '__main__':
    size = 8000
    first_pca = first_pca_component(size)
    last_pca = last_pca_component(size)

    model_evaluation(first_pca, get_distances)
    # model_evaluation(last_pca, get_distances)
