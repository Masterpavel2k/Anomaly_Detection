from mongoDbHb import get_norm_collection, get_anorm_collection
from pca import matrix_construction, pca_matrix_construction
import numpy as np


def err_print(col, pca_comp, title: str):
    err_values = np.zeros(1)
    for hb in col.find({'Test': False}).limit(2):
        s = hb['ML2']
        value = np.matmul(pca_comp, s)
        err_values = np.append(err_values, abs(value))

    err_values = err_values[1:]
    print(title)
    print(err_values)


def first_pca_component(num_hb):
    s_matrix = matrix_construction(num_hb)
    pca = pca_matrix_construction(s_matrix)
    first_pca_comp = pca[:, 1].T
    return first_pca_comp


def last_pca_component(num_hb):
    s_matrix = matrix_construction(num_hb)
    pca = pca_matrix_construction(s_matrix)
    last_pca_comp = pca[:, -1].T
    return last_pca_comp


if __name__ == '__main__':
    size = 4000
    S = matrix_construction(size)
    pca_matrix = pca_matrix_construction(S)
    norm_col = get_norm_collection()
    anorm_col = get_anorm_collection()

    last_PCA_comp = pca_matrix[:, -1].T

    err_print(norm_col, last_PCA_comp, 'Battiti normali ultima componente')
    err_print(anorm_col, last_PCA_comp, 'Battiti anormali ultima componente')

    first_PCA_comp = pca_matrix[:, 1].T
    err_print(norm_col, first_PCA_comp, 'Battiti normali prima componente')
    err_print(anorm_col, first_PCA_comp, 'Battiti anormali prima componente')
