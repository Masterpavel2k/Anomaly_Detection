from random import random
import pandas as pd
import numpy as np
import biosppy.signals as bp


def convert_normal_heart_beats_into_csv(cwd: str = '/Users/paoloberto/PycharmProjects/SignalAnalysis/',
                                        folder_name: str = 'new_normal', file_name: str = 'normal_heart_beats'):
    # the dictionary that contains every heart beat, class and test value
    dataset = {'Class': [], 'Test': []}
    # create the list of string components for the dictionary
    comp_str_cont = []
    for num in range(150):
        comp_str_cont.append('Comp' + str(num))
    # create the list of sampled components
    hb_comp_cont = []
    for el in range(150):
        hb_comp_cont.append([])
    # loop for every csv file
    for num in range(0, 36):
        data = pd.read_csv(cwd + folder_name + '/new_ecg_' + str(num) + '.csv')
        fs = 250
        ts, filtered, r_peaks, templates_ts, templates, hart_rate_ts, hart_rate = bp.ecg.ecg(
            signal=data[data.columns[1]], sampling_rate=fs, show=False)
        # total filtered signal
        filtered = np.array(filtered, dtype='int')
        # loop for every heart beat in the signal
        for peak in r_peaks:
            # set the class to normal
            dataset['Class'].append(0)
            s = filtered[(peak - 51):(peak + 99)]
            # the single heart beat
            new_s = s.tolist()
            for comp in range(150):
                hb_comp_cont[comp].append(new_s[comp])
            # get random number to select test samples
            test_num = random()
            test_bool = 0
            if test_num > 0.8:
                test_bool = 1
            dataset['Test'].append(test_bool)
    # insert heart beats components into dictionary
    for comp_str, comp_list in zip(comp_str_cont, hb_comp_cont):
        dataset[comp_str] = comp_list
    # shuffle and save the dataframe to csv file
    df = pd.DataFrame.from_dict(dataset)
    shuffled = df.sample(frac=1)
    shuffled.to_csv(file_name + '.csv', index=False, header=False)


def convert_anormal_heart_beats_into_csv(cwd: str = '/Users/paoloberto/PycharmProjects/SignalAnalysis/',
                                         folder_name: str = 'new_anormal', file_name: str = 'anormal_heart_beats'):
    # the dictionary that contains every heart beat, class and test value
    dataset = {'Class': [], 'Test': []}
    # create the list of string components for the dictionary
    comp_str_cont = []
    for num in range(150):
        comp_str_cont.append('Comp' + str(num))
    # create the list of sampled components
    hb_comp_cont = []
    for el in range(150):
        hb_comp_cont.append([])
    # loop for every csv file
    for num in range(0, 10):
        for sub in range(0, 5):
            data = pd.read_csv(cwd + folder_name + '/new_an_ecg_' + str(num) + '_' + str(
                sub) + '.csv')
            fs = 360
            ts, filtered, r_peaks, templates_ts, templates, hart_rate_ts, hart_rate = bp.ecg.ecg(
                signal=data[data.columns[1]], sampling_rate=fs, show=False)
            # total filtered signal
            filtered = np.array(filtered, dtype='int')
            # loop for every heart beat in the signal
            for peak in r_peaks:
                # set the class to normal
                dataset['Class'].append(1)
                s = filtered[(peak - 51):(peak + 99)]
                # the single heart beat
                new_s = s.tolist()
                for comp in range(150):
                    hb_comp_cont[comp].append(new_s[comp])
                # get random number to select test samples
                test_num = random()
                test_bool = 0
                if test_num > 0.8:
                    test_bool = 1
                dataset['Test'].append(test_bool)
    # insert heart beats components into dictionary
    for comp_str, comp_list in zip(comp_str_cont, hb_comp_cont):
        dataset[comp_str] = comp_list
    # shuffle and save the dataframe to csv file
    df = pd.DataFrame.from_dict(dataset)
    shuffled = df.sample(frac=1)
    shuffled.to_csv(file_name + '.csv', index=False)


def load_heart_beats_from_csv(file_name: str = 'normal_heart_beats'):
    # load dataframe
    dataset = pd.read_csv(file_name + '.csv')
    # select train and test samples
    dataset_train = dataset.loc[dataset['Test'] == 0]
    dataset_test = dataset.loc[dataset['Test'] == 1]
    # remove unnecessary column
    train_array = dataset_train.to_numpy()
    train_array = np.delete(train_array, 1, 1)
    test_array = dataset_test.to_numpy()
    test_array = np.delete(test_array, 1, 1)
    return train_array, test_array


if __name__ == '__main__':
    convert_anormal_heart_beats_into_csv()
    train_array, test_array = load_heart_beats_from_csv('anormal_heart_beats')
    print(train_array, test_array)
