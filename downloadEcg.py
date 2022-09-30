import os
import subprocess


def needs_download(cwd: str, folder_name: str):
    os.chdir(cwd)
    if not os.path.exists(cwd + folder_name):
        os.mkdir(cwd + folder_name)
        return True
    elif not os.listdir(cwd + folder_name):
        return True
    return False


def download_normal_ecg(cwd: str, folder_name: str, patient_number: str):
    for num in range(0, 36):
        f_out = open(cwd + folder_name + '/raw_ecg_' +
                     str(num) + '.csv', 'wb')
        start_time = num * 300
        end_time = (num + 1) * 300
        subprocess.run(
            ['rdsamp', '-r', 'ltstdb/' + patient_number, '-c', '-H', '-f', str(start_time), '-t', str(end_time), '-v',
             '-ps'],
            stdout=f_out)
    print('Download done!')


def download_abnormal_ecg(cwd: str, folder_name: str, patient_number: str):
    for num in range(0, 10):
        for sub in range(0, 5):
            fout = open(cwd + folder_name + '/raw_abn_ecg_' + str(num) + '_' + str(sub) + '.csv', 'wb')
            start_time = sub * 300
            end_time = (sub + 1) * 300
            db_name = 'mitdb/' + patient_number + str(num)
            subprocess.run(
                ['rdsamp', '-r', db_name, '-c', '-H', '-f', str(start_time), '-t', str(end_time), '-v', '-ps'],
                stdout=fout)
    print('Download done!')


def download_ecg(cwd: str, normal_folder_name: str, abnormal_folder_name: str):
    normal_patient = 's20031'
    if needs_download(cwd, normal_folder_name):
        download_normal_ecg(cwd, normal_folder_name, normal_patient)
    abnormal_patient = '10'
    if needs_download(cwd, abnormal_folder_name):
        download_abnormal_ecg(cwd, abnormal_folder_name, abnormal_patient)
