import numpy as np
import pymongo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

myclient = pymongo.MongoClient('mongodb://masterpavel:5uunt0@192.168.1.125:27017/?authMechanism=DEFAULT')
mydb = myclient['SignalAnalysis']
mycol = mydb['HeartBeats']

query = {'Test': False}

normal_pointer = mycol.find(query).limit(10)
coll_norm = np.zeros(1)
for el in normal_pointer:
    coll_norm = np.append(coll_norm, el['Class'])

myanomcol = mydb['AnomHeartBeats']
anormal_pointer = myanomcol.find(query).limit(10)
coll_anorm = np.zeros(1)
for el in anormal_pointer:
    coll_anorm = np.append(coll_anorm, el['Class'])

print(coll_norm)
print(coll_anorm)

y_true = ['Normal', 'Anormal', 'Normal', 'Anormal']
y_pred = ['Normal', 'Anormal', 'Anormal', 'Normal']

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(tn, fp, fn, tp)

y = np.array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1])
scores = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9])
fpr, tpr, thresholds = roc_curve(y, scores)
print(fpr, tpr, thresholds)
auc_value = auc(fpr, tpr)
print(auc_value)
