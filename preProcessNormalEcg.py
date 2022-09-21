import subprocess
import os

import pandas as pd


def pre_process_normal_ecg(folder_name: str, cwd: str = '/Users/paoloberto/PycharmProjects/SignalAnalysis/'):
    new_folder_name = folder_name + '_processed'
    os.chdir(cwd)
    subprocess.run(['mkdir', new_folder_name])
    for num in range(0, 36):
        raw_data = pd.read_csv(cwd + folder_name + '/raw_ecg_' + str(num) + '.csv',
                               sep=',')
        new_data = raw_data.loc[1:, [raw_data.columns[0], raw_data.columns[1]]]
        new_data = new_data.astype('float64')
        new_data = new_data * [1, 1000]
        new_data = new_data.astype({new_data.columns[1]: 'int32'})
        new_data.to_csv(cwd + new_folder_name + '/new_ecg_' + str(num) + '.csv',
                        index=False)
    print('Done!')
    return new_folder_name
