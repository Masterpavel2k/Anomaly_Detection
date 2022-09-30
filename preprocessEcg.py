import os

import pandas as pd


def needs_preprocess(cwd: str, folder_name: str):
    os.chdir(cwd)
    if not os.path.exists(cwd + folder_name):
        os.mkdir(cwd + folder_name)
        return True
    elif not os.listdir(cwd + folder_name):
        return True
    return False


def preprocess_normal_ecg(cwd: str, folder_name: str, new_folder_name: str):
    for num in range(0, 36):
        raw_data = pd.read_csv(cwd + folder_name + '/raw_ecg_' + str(num) + '.csv',
                               sep=',')
        new_data = raw_data.loc[1:, [raw_data.columns[0], raw_data.columns[1]]]
        new_data = new_data.astype('float64')
        new_data = new_data * [1, 1000]
        new_data = new_data.astype({new_data.columns[1]: 'int32'})
        new_data.to_csv(cwd + new_folder_name + '/pro_ecg_' + str(num) + '.csv',
                        index=False)
    print('Preprocessing done!')


def preprocess_abnormal_ecg(cwd: str, folder_name: str, new_folder_name: str):
    for num in range(0, 10):
        for sub in range(0, 5):
            raw_data = pd.read_csv(cwd + folder_name + '/raw_abn_ecg_' +
                                   str(num) + '_' + str(sub) + '.csv', sep=',')
            new_data = raw_data.loc[1:, [raw_data.columns[0], raw_data.columns[1]]]
            new_data = new_data.astype('float64')
            new_data = new_data * [1, 1000]
            new_data = new_data.astype({new_data.columns[1]: 'int32'})
            new_data.to_csv(cwd + new_folder_name + '/pro_abn_ecg_' +
                            str(num) + '_' + str(sub) + '.csv', index=False)
    print('Preprocessing done!')


def preprocess_ecg(cwd: str, normal_folder_name: str, abnormal_folder_name: str):
    new_normal_folder_name = normal_folder_name + '_processed'
    new_abnormal_folder_name = abnormal_folder_name + '_processed'
    if needs_preprocess(cwd, new_normal_folder_name):
        preprocess_normal_ecg(cwd, normal_folder_name, new_normal_folder_name)
    if needs_preprocess(cwd, new_abnormal_folder_name):
        preprocess_abnormal_ecg(cwd, abnormal_folder_name, new_abnormal_folder_name)
    return new_normal_folder_name, new_abnormal_folder_name
