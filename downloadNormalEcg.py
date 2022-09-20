import subprocess

for num in range(0, 36):
    fout = open('/Users/paoloberto/PycharmProjects/SignalAnalysis/raw_normal/raw_ecg_' + str(num) + '.csv', 'wb')
    start_time = num * 300
    end_time = (num + 1) * 300
    subprocess.run(
        ['rdsamp', '-r', 'ltstdb/s20011', '-c', '-H', '-f', str(start_time), '-t', str(end_time), '-v', '-ps'],
        stdout=fout)

# fout = open('samples2.csv', 'wb')
# subprocess.run(['rdsamp', '-r', 'ltstdb/s20011', '-c', '-H', '-f', '0', '-t', '300', '-v', '-pd'], stdout=fout)

print('Done!')
