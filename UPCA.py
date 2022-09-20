from matplotlib import pyplot as plt
from PCA import matrix_construction
import numpy as np
import pymongo


def err_print(dbcol, PCA_comp, title: str):
    err_values = np.zeros(1)
    num_of_hb = 10
    for hb in dbcol.find({'Test': False}):
        s = hb['ML2']
        num_of_hb = num_of_hb - 1
        if num_of_hb == 0:
            break
        value = np.matmul(PCA_comp, s)
        err_values = np.append(err_values, value)

    err_values = err_values[1:]
    print(title)
    print(err_values)


def err_anom_print(comp, title: str):
    anomDbCol = mydb['AnomHeartBeats']
    err_print(anomDbCol, comp, 'Battiti anomali ' + title)


if __name__ == '__main__':
    myclient = pymongo.MongoClient('mongodb://masterpavel:5uunt0@192.168.1.125:27017/?authMechanism=DEFAULT')
    mydb = myclient['SignalAnalysis']
    mycol = mydb['HeartBeats']

    size = 10000
    S = matrix_construction(size)
    U, D, Vt = np.linalg.svd(np.matmul(S, S.T))
    print(U.shape)

    last_PCA_comp = U[:, -1].T
    # print('last PCA')
    # print(last_PCA_comp)

    first_PCA_comp = U[:, 1].T
    # print('first PCA')
    # print(first_PCA_comp)

    err_print(mycol, first_PCA_comp, 'Battiti Normali Prima PCA')
    err_print(mycol, last_PCA_comp, 'Battiti Normali Ultima PCA')

    err_anom_print(first_PCA_comp, 'Prima PCA')
    err_anom_print(last_PCA_comp, 'Ultima PCA')

"""
im = plt.imshow(U, cmap="copper_r")
plt.colorbar(im)
plt.show()
"""
