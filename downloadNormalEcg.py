import subprocess
import os
from databaseNormalPopulation import populate_db_with_norm_hb
from preProcessNormalEcg import pre_process_normal_ecg


def download_norm_ecg(folder_name: str, patient_number: str, col_name: str = 'HeartBeats',
                      cwd: str = '/Users/paoloberto/PycharmProjects/SignalAnalysis/'):
    os.chdir(cwd)
    subprocess.run(['mkdir', folder_name])
    for num in range(0, 36):
        f_out = open(cwd + folder_name + '/raw_ecg_' +
                     str(num) + '.csv', 'wb')
        start_time = num * 300
        end_time = (num + 1) * 300
        subprocess.run(
            ['rdsamp', '-r', 'ltstdb/' + patient_number, '-c', '-H', '-f', str(start_time), '-t', str(end_time), '-v',
             '-ps'],
            stdout=f_out)
        print(str(num) + ' Done!')
    print('Download done!')
    new_folder_name = pre_process_normal_ecg(folder_name, cwd)
    print('Pre process done!')
    populate_db_with_norm_hb(new_folder_name, cwd, col_name)
