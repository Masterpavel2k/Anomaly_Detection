import numpy as np


def average_heart_beat(heart_beats):
    matrix = np.zeros(150)
    for hb in heart_beats:
        matrix = np.vstack((matrix, hb))
    matrix = matrix[1:]
    matrix = np.array(matrix, dtype='int')
    avg = np.average(matrix, axis=0)
    avg = np.array(avg, dtype='int')
    return avg


def get_distances(average_hb, heart_beats):
    dist_coll = np.zeros(1)
    for hb in heart_beats:
        dist = np.linalg.norm(average_hb - hb)
        dist_coll = np.append(dist_coll, dist)
    dist_coll = dist_coll[1:]
    dist_coll = dist_coll.reshape(-1, 1)
    return dist_coll
