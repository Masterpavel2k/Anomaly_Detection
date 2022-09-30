import numpy as np
import pandas as pd


def load_heart_beats_from_csv(file_name: str):
    # load dataframe
    dataset = pd.read_csv(file_name)
    # select train and test samples
    dataset_train = dataset.loc[dataset['Test'] == 0]
    dataset_test = dataset.loc[dataset['Test'] == 1]
    # remove unnecessary column
    train_array = dataset_train.to_numpy()
    train_array = np.delete(train_array, 1, 1)
    test_array = dataset_test.to_numpy()
    test_array = np.delete(test_array, 1, 1)
    return train_array, test_array


def get_sets(size: int, cwd: str, normal_file_name: str, abnormal_file_name: str):
    # get the normal and abnormal train and test datasets
    normal_train_ds, normal_test_ds = load_heart_beats_from_csv(cwd + '/' + normal_file_name)
    abnormal_train_ds, abnormal_test_ds = load_heart_beats_from_csv(cwd + '/' + abnormal_file_name)
    # division in train / test and heart beats / classes
    train_hb = np.append(normal_train_ds[:size, 1:], abnormal_train_ds[:size, 1:], axis=0)
    train_cls = np.append(normal_train_ds[:size, 0], abnormal_train_ds[:size, 0], axis=0)
    test_hb = np.append(normal_test_ds[:size, 1:], abnormal_test_ds[:size, 1:], axis=0)
    test_cls = np.append(normal_test_ds[:size, 0], abnormal_test_ds[:size, 0], axis=0)
    return train_hb, train_cls, test_hb, test_cls
