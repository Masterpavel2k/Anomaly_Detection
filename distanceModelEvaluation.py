from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from timer import timer


def get_log_model(given_model, get_distances, train_hb, train_cls):
    # get the model specific distances
    train_distances = get_distances(given_model, train_hb)
    # training of linear regression model
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(train_distances, train_cls)
    return model


def new_model_evaluation(given_model, model_name: str, get_distances, train_heart_beats, train_classes,
                         test_heart_beats, test_classes):
    # get the model specific distances
    train_distances = get_distances(given_model, train_heart_beats)
    # training of linear regression model
    with timer(name_or_func='log_reg'):
        model = LogisticRegression(solver='liblinear', C=10.0, random_state=0).fit(train_distances, train_classes)
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
    display_curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name + ' + Log Reg')
    display_curve.plot()
    plt.show()
