import subprocess

for num in range(0, 10):
    for sub in range(0, 5):
        fout = open('/Users/paoloberto/PycharmProjects/SignalAnalysis/raw_anormal/raw_an_ecg_' + str(num) + '_' + str(
            sub) + '.csv', 'wb')
        start_time = sub * 300
        end_time = (sub + 1) * 300
        db_name = 'mitdb/10' + str(num)
        subprocess.run(['rdsamp', '-r', db_name, '-c', '-H', '-f', str(start_time), '-t', str(end_time), '-v', '-ps'],
                       stdout=fout)

print('Done!')
