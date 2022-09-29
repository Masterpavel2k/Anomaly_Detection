import os

from convertEcgInCsv import convert_ecg_in_csv, load_heart_beats_from_csv
from downloadEcg import download_ecg
from preprocessEcg import preprocess_ecg

if __name__ == '__main__':
    cwd = os.getcwd()
    normal_folder_name = '/normal'
    abnormal_folder_name = '/abnormal'
    normal_file_name = 'normalHeartBeats.csv'
    abnormal_file_name = 'abnormalHeartBeats.csv'

    download_ecg(cwd, normal_folder_name, abnormal_folder_name)
    new_normal_folder, new_abnormal_folder = preprocess_ecg(cwd, normal_folder_name, abnormal_folder_name)
    convert_ecg_in_csv(cwd, new_normal_folder, new_abnormal_folder, normal_file_name, abnormal_file_name)
    train_ds, test_ds = load_heart_beats_from_csv(cwd + '/' + normal_file_name)
    print(train_ds, test_ds)

    """
    model_container = []
    for model in model_container:
        train(model)
        evaluate(model)
    """
