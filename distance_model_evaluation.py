import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay

from heartBeatsExtraction import get_train_test_heart_beats, get_train_test_from_collection


def model_evaluation(given_model, get_distances):
    # number of heart beats to be used for logistic regression
    log_reg_hb_size = 5000
    # get train and test heart beats and classes
    train_heart_beats, train_classes, test_heart_beats, test_classes = get_train_test_heart_beats(log_reg_hb_size)
    # get the model specific distances
    train_distances = get_distances(given_model, train_heart_beats)
    # training of linear regression model
    print('Starting fitting')
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(train_distances, train_classes)
    print('Finished fitting')
    # get the model specific distances
    test_distances = get_distances(given_model, test_heart_beats)
    # accuracy metrics
    accuracy = model.score(test_distances, test_classes)
    print('Accuracy: ' + str(accuracy))
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


# experimental, just for test purposes
def model_test(given_model, get_distances, col_name: str, log_reg_size: int = 10):
    train_hb, train_cls, test_hb, test_cls = get_train_test_from_collection(log_reg_size, col_name)
    train_distances = get_distances(given_model, train_hb)
    return train_distances


# used to compare prediction against ecg from another person
def generic_model_evaluation(given_model, get_distances, col_name: str):
    log_reg_hb_size = 5000

    train_heart_beats, train_classes, test_heart_beats, test_classes = get_train_test_heart_beats(log_reg_hb_size)

    train_distances = get_distances(given_model, train_heart_beats)

    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(train_distances, train_classes)

    size = 100
    test_distances = model_test(given_model, get_distances, col_name, size)
    test_cls = np.zeros(size).astype(dtype='int')
    ans = model.predict(test_distances)
    conf_m = confusion_matrix(test_cls, ans)
    print(conf_m)
