import numpy as np
import pandas as pd
from mongoDbHb import get_norm_collection, get_anorm_collection
from matplotlib import pyplot as plt
import seaborn as sns


def matrix_construction(size: int):
    col = get_norm_collection()
    S = np.zeros(150)
    for nb in col.find({'Test': False}).limit(size):
        s = nb['ML2']
        S = np.vstack((S, s))

    S = S[1:]
    S = np.array(S, dtype='int')
    S = S.T
    return S


def avg_err_norm(U, size: int):
    col = get_norm_collection()
    k_values = np.arange(1, U.shape[0])
    err_values = np.zeros(1)
    for k in k_values:
        Uk = U[:, :k]
        UUt = np.matmul(Uk, Uk.T)
        err = np.zeros(1)
        for nb in col.find({'Test': False}).limit(size):
            s = nb['ML2']
            samp_err = np.linalg.norm(s - (np.matmul(UUt, s)))
            err = np.vstack((err, samp_err))
        err = err[1:]
        avg_err = np.average(err)
        err_values = np.append(err_values, avg_err)

    err_values = err_values[1:]
    sum_values = pd.DataFrame({'k': k_values, 'average error': err_values})
    return sum_values


def avg_err_anom(U, size: int):
    col = get_anorm_collection()
    k_values = np.arange(1, U.shape[0])
    err_values = np.zeros(1)
    for k in k_values:
        Uk = U[:, :k]
        UUt = np.matmul(Uk, Uk.T)
        err = np.zeros(1)
        for nb in col.find({'Test': False}).limit(size):
            s = nb['ML2']
            samp_err = np.linalg.norm(s - (np.matmul(UUt, s)))
            err = np.vstack((err, samp_err))
        err = err[1:]
        avg_err = np.average(err)
        err_values = np.append(err_values, avg_err)

    err_values = err_values[1:]
    sum_values = pd.DataFrame({'k': k_values, 'average error': err_values})
    return sum_values


def pca_matrix_construction(matrix):
    U, D, Vt = np.linalg.svd(np.matmul(matrix, matrix.T))
    return U


def norm_from_optimal_pca(pca_matrix, k, heart_beat):
    Uk = pca_matrix[:, :k]
    UUt = np.matmul(Uk, Uk.T)
    distance = np.linalg.norm(heart_beat - (np.matmul(UUt, heart_beat)))
    return distance


def example():
    size = 10000
    S = matrix_construction(size)
    U, D, Vt = np.linalg.svd(np.matmul(S, S.T))
    print(U.shape)
    dframe = avg_err_norm(U, int(size / 100))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=dframe[dframe.columns[1]])
    # per k = 40 errore = 80
    dframe_anom = avg_err_anom(U, int(size / 100))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=dframe_anom[dframe_anom.columns[1]])
    # per k = 40 errore = 300
    plt.show()
