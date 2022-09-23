import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from datasetExperiment import from_csv_to_dataset
from heartBeatsExtraction import get_train_test_heart_beats


def evaluate(pipe_lr, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=100, n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.87, 1.005])
    plt.show()


if __name__ == '__main__':
    size = 10000
    file_name = 'new_prova'
    # train_hb, train_cls, test_hb, test_cls = from_csv_to_dataset(size, file_name)
    train_hb, train_cls, test_hb, test_cls = get_train_test_heart_beats(size)
    pipe_lr = make_pipeline(StandardScaler(), PCA(n_components=2), LogisticRegression(random_state=1))

    evaluate(pipe_lr, train_hb, train_cls)
