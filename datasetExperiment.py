import os.path

from heartBeatsExtraction import get_train_test_heart_beats
import pandas as pd
import numpy as np


def csv_creation(heart_beats, classes, file_name: str):
    list_container = []
    for comp in range(150):
        comp_list = []
        for hb in heart_beats:
            comp_list.append(hb[comp])
        list_container.append(comp_list)

    comp_str_cont = []
    for num in range(150):
        comp_str_cont.append('Comp' + str(num))

    data = {'Class': classes}
    for comp_str, comp_list in zip(comp_str_cont, list_container):
        data[comp_str] = comp_list

    df = pd.DataFrame.from_dict(data)
    shuffled = df.sample(frac=1)
    shuffled.to_csv(file_name + '.csv', index=False, header=False)


def from_db_to_csv(size: int, file_name: str):
    train_hb, train_cls, test_hb, test_cls = get_train_test_heart_beats(size)
    csv_creation(train_hb, train_cls, file_name)
    csv_creation(test_hb, test_cls, file_name + '_test')


def from_csv_to_dataset(size: int, file_name: str):
    if not os.path.exists(file_name):
        from_db_to_csv(size, file_name)
    train_data = np.loadtxt(file_name + '.csv', delimiter=',').astype(dtype='int')
    test_data = np.loadtxt(file_name + '_test' + '.csv', delimiter=',').astype(dtype='int')
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]
    x_test = test_data[:, 1:]
    y_test = test_data[:, 0]
    return x_train, y_train, x_test, y_test
