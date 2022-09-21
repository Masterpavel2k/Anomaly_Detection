from downloadNormalEcg import download_norm_ecg


def download():
    folder = 'nuovo_paziente'
    col_name = 'HeartBeats2'
    patient_number = 's20031'
    download_norm_ecg(folder, patient_number, col_name)
