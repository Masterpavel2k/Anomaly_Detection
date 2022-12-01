from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from timer import timer
import logging


logging.basicConfig(level=logging.INFO)
timer.set_level(logging.INFO)


def pipeline_model(train_hb, train_cls, test_hb, test_cls):
    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))
    with timer():
        pipe_lr.fit(train_hb, train_cls)
    print('Best Model')
    accuracy = pipe_lr.score(test_hb, test_cls)
    print('Accuracy: ' + str(accuracy))

    ans = pipe_lr.predict(test_hb)
    conf_m = confusion_matrix(test_cls, ans)
    print(conf_m)

    prob_ans = pipe_lr.predict_proba(test_hb)
    pos_prob_ans = prob_ans[:, 1]
    fpr, tpr, thresholds = roc_curve(test_cls, pos_prob_ans)
    roc_auc = auc(fpr, tpr)
    print('Auc Roc: ' + str(roc_auc))
    # roc curve plotting and display
    display_curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Scaler + PCA + Log Reg')
    display_curve.plot()
    plt.show()
