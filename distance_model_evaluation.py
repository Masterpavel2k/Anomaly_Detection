import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay

from heartBeatsExtraction import get_heart_beats
from mongoDbHb import get_norm_collection, get_anorm_collection


def model_evaluation(given_model, get_distances):
    # number of heart beats to be used for logistic regression
    log_reg_hb_size = 5000
    # normal heart beats collection from database
    norm_col = get_norm_collection()
    # split heart beats in different classes
    norm_train_heart_beats, norm_train_classes, norm_test_heart_beats, norm_test_classes = get_heart_beats(
        norm_col, log_reg_hb_size)
    # anormal heart beats collection from database
    anorm_col = get_anorm_collection()
    # split heart beats in different classes
    anorm_train_heart_beats, anorm_train_classes, anorm_test_heart_beats, anorm_test_classes = get_heart_beats(
        anorm_col, log_reg_hb_size)
    # subdivision of heart beats
    train_heart_beats = np.vstack((norm_train_heart_beats, anorm_train_heart_beats))
    train_classes = np.append(norm_train_classes, anorm_train_classes)
    train_distances = get_distances(given_model, train_heart_beats)
    # training of linear regression model
    print('Starting fitting')
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(train_distances, train_classes)
    print('Finished fitting')
    # subdivision of heart beats
    test_heart_beats = np.vstack((norm_test_heart_beats, anorm_test_heart_beats))
    test_classes = np.append(norm_test_classes, anorm_test_classes)
    test_distances = get_distances(given_model, test_heart_beats)
    # prediction of the model with confusion matrix
    ans = model.predict(test_distances)
    conf_m = confusion_matrix(test_classes, ans)
    print(conf_m)
    # probability of prediction for roc curve
    prob_ans = model.predict_proba(test_distances)
    pos_prob_ans = prob_ans[:, 1]
    fpr, tpr, thresholds = roc_curve(test_classes, pos_prob_ans)
    roc_auc = auc(fpr, tpr)
    # roc curve plotting and display
    display_curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='PCA + Log Reg')
    display_curve.plot()
    plt.show()
