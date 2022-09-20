import numpy as np
import pandas as pd
import biosppy.signals as bp
import pymongo
import random

myclient = pymongo.MongoClient('mongodb://masterpavel:5uunt0@192.168.1.125:27017/?authMechanism=DEFAULT')
mydb = myclient['SignalAnalysis']
mycol = mydb['AnomHeartBeats']

for num in range(0, 10):
    for sub in range(0, 5):
        data = pd.read_csv(
            '/Users/paoloberto/PycharmProjects/SignalAnalysis/new_anormal/new_an_ecg_' + str(num) + '_' + str(
                sub) + '.csv')
        fs = 360
        ts, filtered, r_peaks, tamplates_ts, tamplates, hart_rate_ts, hart_rate = bp.ecg.ecg(
            signal=data[data.columns[1]], sampling_rate=fs, show=False)
        filtered = np.array(filtered, dtype='int')
        for peak in r_peaks:
            s = filtered[(peak - 51):(peak + 99)]
            new_s = s.tolist()
            test_num = random.random()
            test_bool = False
            if test_num > 0.8:
                test_bool = True
            heartbeat = {
                "ML2": new_s,
                "Class": "Anormal",
                "Method Prediction": [],
                "Test": test_bool
            }
            id = mycol.insert_one(heartbeat)
        print("Done " + str(num))
