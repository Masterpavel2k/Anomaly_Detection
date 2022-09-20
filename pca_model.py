import numpy as np
from mongoDbHb import get_norm_collection, get_anorm_collection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
from PCA import matrix_construction, norm_from_optimal_pca, pca_matrix_construction


def get_arrays(coll):
    heart_beats = np.zeros(150)
    classes = np.zeros(1)
    for hb in coll:
        heart_beats = np.vstack((heart_beats, hb['ML2']))
        classes = np.append(classes, (0 if hb['Class'] == 'Normal' else 1))

    heart_beats = heart_beats[1:].astype(dtype='int')
    classes = classes[1:].astype(dtype='int')

    return heart_beats, classes


def get_heart_beats(dbcol, num_of_hb: int):
    train_coll = dbcol.find({'Test': False}).limit(num_of_hb)
    test_coll = dbcol.find({'Test': True}).limit(num_of_hb)

    train_heart_beats_local, train_classes_local = get_arrays(train_coll)
    test_heart_beats_local, test_classes_local = get_arrays(test_coll)

    return train_heart_beats_local, train_classes_local, test_heart_beats_local, test_classes_local


def get_distances(matrix, heart_beats):
    dist_coll = np.zeros(1)
    for hb in heart_beats:
        dist = norm_from_optimal_pca(matrix, 50, hb)
        dist_coll = np.append(dist_coll, dist)
    dist_coll = dist_coll[1:]
    dist_coll = dist_coll.reshape(-1, 1)
    return dist_coll


if __name__ == '__main__':
    s_matrix = matrix_construction(4000)
    pca_matrix = pca_matrix_construction(s_matrix)
    log_reg_hb_size = 1000

    norm_col = get_norm_collection()

    norm_train_heart_beats, norm_train_classes, norm_test_heart_beats, norm_test_classes = get_heart_beats(norm_col,
                                                                                                           log_reg_hb_size)

    anorm_col = get_anorm_collection()

    anorm_train_heart_beats, anorm_train_classes, anorm_test_heart_beats, anorm_test_classes = get_heart_beats(
        anorm_col, log_reg_hb_size)

    train_heart_beats = np.vstack((norm_train_heart_beats, anorm_train_heart_beats))
    train_classes = np.append(norm_train_classes, anorm_train_classes)
    train_distances = get_distances(pca_matrix, train_heart_beats)

    print('Starting fitting')
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(train_distances, train_classes)
    print('Finished fitting')

    test_heart_beats = np.vstack((norm_test_heart_beats, anorm_test_heart_beats))
    test_classes = np.append(norm_test_classes, anorm_test_classes)
    test_distances = get_distances(pca_matrix, test_heart_beats)

    ans = model.predict(test_distances)
    conf_m = confusion_matrix(test_classes, ans)
    print(conf_m)

    prob_ans = model.predict_proba(test_distances)
    pos_prob_ans = prob_ans[:, 1]
    fpr, tpr, thresholds = roc_curve(test_classes, pos_prob_ans)
    roc_auc = auc(fpr, tpr)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='PCA + Log Reg')
    display.plot()
    plt.show()
