import numpy as np
from averageHB import avg_hb_construction
from distance_model_evaluation import model_evaluation


def get_distances(average_hb, heart_beats):
    dist_coll = np.zeros(1)
    for hb in heart_beats:
        dist = np.linalg.norm(average_hb - hb)
        dist_coll = np.append(dist_coll, dist)
    dist_coll = dist_coll[1:]
    dist_coll = dist_coll.reshape(-1, 1)
    return dist_coll


if __name__ == '__main__':
    # construction of average heart beat, parameter is the number of heart beats
    avg_hb = avg_hb_construction(4000)

    model_evaluation(avg_hb, get_distances)
