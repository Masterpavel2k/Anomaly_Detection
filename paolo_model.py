from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from datasetExperiment import from_csv_to_dataset
from heartBeatsExtraction import get_train_test_heart_beats

if __name__ == '__main__':
    # train_hb, train_cls, test_hb, test_cls = get_train_test_heart_beats(5000)
    size = 5000
    file_name = 'new_prova'
    train_hb, train_cls, test_hb, test_cls = from_csv_to_dataset(size, file_name)

    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))
    print('Start')
    pipe_lr.fit(train_hb, train_cls)
    print('Finish')
    accuracy = pipe_lr.score(test_hb, test_cls)
    print('Accuracy: ' + str(accuracy))

    ans = pipe_lr.predict(test_hb)
    conf_m = confusion_matrix(test_cls, ans)
    print(conf_m)

    prob_ans = pipe_lr.predict_proba(test_hb)
    pos_prob_ans = prob_ans[:, 1]
    fpr, tpr, thresholds = roc_curve(test_cls, pos_prob_ans)
    roc_auc = auc(fpr, tpr)
    # roc curve plotting and display
    display_curve = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Scaler + PCA + Log Reg')
    display_curve.plot()
    plt.show()
