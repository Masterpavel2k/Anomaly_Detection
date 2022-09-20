import numpy as np


def get_arrays(coll):
    heart_beats = np.zeros(150)
    classes = np.zeros(1)
    for hb in coll:
        heart_beats = np.vstack((heart_beats, hb['ML2']))
        classes = np.append(classes, (0 if hb['Class'] == 'Normal' else 1))

    heart_beats = heart_beats[1:].astype(dtype='int')
    classes = classes[1:].astype(dtype='int')

    return heart_beats, classes


def get_heart_beats(col, num_of_hb: int):
    train_coll = col.find({'Test': False}).limit(num_of_hb)
    test_coll = col.find({'Test': True}).limit(num_of_hb)

    train_heart_beats_local, train_classes_local = get_arrays(train_coll)
    test_heart_beats_local, test_classes_local = get_arrays(test_coll)

    return train_heart_beats_local, train_classes_local, test_heart_beats_local, test_classes_local
