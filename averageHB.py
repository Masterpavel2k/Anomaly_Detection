import numpy as np
from mongoDbHb import get_anorm_collection, get_norm_collection
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def avg_hb(size: int, col):
    matrix = np.zeros(150)
    for nb in col.find({'Test': False}).limit(size):
        matrix = np.vstack((matrix, nb['ML2']))

    matrix = matrix[1:]
    matrix = np.array(matrix, dtype='int')

    avg = np.average(matrix, axis=0)
    avg = np.array(avg, dtype='int')
    return avg


def avg_comparison(col):
    iterations = [1000, 5000, 10000]
    components = range(0, 150)
    dframe = pd.DataFrame({'component': components})
    dframe['first'] = avg_hb(iterations[0], col)
    dframe['second'] = avg_hb(iterations[1], col)
    dframe['third'] = avg_hb(iterations[2], col)

    dframe_2 = pd.melt(dframe, 'component', var_name='Average', value_name='Value')
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot('component', 'Value', hue='Average', data=dframe_2)
    # poche differenze tra 1000 e 10000, incertezza nella coda del battito
    plt.show()


def err_print(dbcol, heart_beat, title: str):
    err_values = np.zeros(1)
    num_of_hb = 10
    for hb in dbcol.find({'Test': False}).limit(num_of_hb):
        s = hb['ML2']
        value = np.linalg.norm(heart_beat - s)
        err_values = np.append(err_values, value)

    err_values = err_values[1:]
    print(title)
    print(err_values)


if __name__ == '__main__':
    norm_col = get_norm_collection()
    average_heart_beat = avg_hb(5000, norm_col)

    err_print(norm_col, average_heart_beat, 'Battito Medio Battiti Normali')
    anorm_col = get_anorm_collection()
    err_print(anorm_col, average_heart_beat, 'Battito Medio Battiti Anormali')
