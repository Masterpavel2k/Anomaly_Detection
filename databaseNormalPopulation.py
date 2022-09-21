import numpy as np
import pandas as pd
import biosppy.signals as bp
import random

from mongoDbHb import get_client


def populate_db_with_norm_hb(folder_name: str, cwd: str = '/Users/paoloberto/PycharmProjects/SignalAnalysis/',
                             col_name: str = 'HeartBeats'):
    db = get_client()
    col = db[col_name]
    for num in range(0, 36):
        data = pd.read_csv(cwd + folder_name + '/new_ecg_' +
                           str(num) + '.csv')
        fs = 250
        ts, filtered, r_peaks, templates_ts, templates, hart_rate_ts, hart_rate = bp.ecg.ecg(
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
                "Class": "Normal",
                "Method Prediction": [],
                "Test": test_bool
            }
            col.insert_one(heartbeat)
        print("Done " + str(num))
