import pandas as pd

for num in range(0, 10):
    for sub in range(0, 5):
        raw_data = pd.read_csv(
            '/Users/paoloberto/PycharmProjects/SignalAnalysis/raw_anormal/raw_an_ecg_' + str(num) + '_' + str(
                sub) + '.csv', sep=',')
        new_data = raw_data.loc[1:, [raw_data.columns[0], raw_data.columns[1]]]
        new_data = new_data.astype('float64')
        new_data = new_data * [1, 1000]
        new_data = new_data.astype({new_data.columns[1]: 'int32'})
        new_data.to_csv(
            '/Users/paoloberto/PycharmProjects/SignalAnalysis/new_anormal/new_an_ecg_' + str(num) + '_' + str(
                sub) + '.csv', index=False)

print('Done!')
