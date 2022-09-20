import numpy as np

from mongoDbHb import get_norm_collection, get_anorm_collection


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


def get_train_test_heart_beats(size: int):
    # normal heart beats collection from database
    norm_col = get_norm_collection()
    # split heart beats in different classes
    norm_train_heart_beats, norm_train_classes, norm_test_heart_beats, norm_test_classes = get_heart_beats(
        norm_col, size)
    # anormal heart beats collection from database
    anorm_col = get_anorm_collection()
    # split heart beats in different classes
    anorm_train_heart_beats, anorm_train_classes, anorm_test_heart_beats, anorm_test_classes = get_heart_beats(
        anorm_col, size)
    # subdivision of heart beats
    train_heart_beats = np.vstack((norm_train_heart_beats, anorm_train_heart_beats))
    train_classes = np.append(norm_train_classes, anorm_train_classes)
    test_heart_beats = np.vstack((norm_test_heart_beats, anorm_test_heart_beats))
    test_classes = np.append(norm_test_classes, anorm_test_classes)
    return train_heart_beats, train_classes, test_heart_beats, test_classes
