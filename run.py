import os
import sys
from avgHbModel import average_heart_beat, get_distances as avg_get_distances
from convertEcgInCsv import convert_ecg_in_csv
from datasetExtraction import get_sets
from distanceModelEvaluation import new_model_evaluation
from downloadEcg import download_ecg
from pcaModel import get_pca_matrix, get_distances as pca_get_distances, get_distances_50, get_distances_40, \
    get_distances_20, get_distances_100
from pipelineModel import pipeline_model
from preprocessEcg import preprocess_ecg
from lastPcaModel import first_pca_component, last_pca_component, get_distances as last_pca_get_distances

if __name__ == '__main__':
    # command line parameters
    args = sys.argv[1:]
    # names and parameters
    cwd = os.getcwd()
    normal_folder_name = '/normal'
    abnormal_folder_name = '/abnormal'
    normal_file_name = 'normalHeartBeats.csv'
    abnormal_file_name = 'abnormalHeartBeats.csv'
    size = 5000
    # download, preprocess and convert ecg if necessary
    download_ecg(cwd, normal_folder_name, abnormal_folder_name)
    new_normal_folder, new_abnormal_folder = preprocess_ecg(cwd, normal_folder_name, abnormal_folder_name)
    convert_ecg_in_csv(cwd, new_normal_folder, new_abnormal_folder, normal_file_name, abnormal_file_name)
    # start models comparison
    train_hb, train_cls, test_hb, test_cls = get_sets(size, cwd, normal_file_name, abnormal_file_name)
    print('Number of train samples ', len(train_hb), ' Number of test classes ', len(test_cls))
    # setup of arrays
    distance_models = []
    models_names = []
    distance_functions = []
    pipeline_model_select = False
    # selection of models to evaluate
    for arg in args:
        if arg == 'compare-all':
            print('Comparison of all methods')
        elif arg == 'average-heartbeat':
            print('Evaluation of average heartbeat method')
            distance_models.append(average_heart_beat(train_hb))
            models_names.append('Average HeartBeat')
            distance_functions.append(avg_get_distances)
        elif arg == 'last-pca':
            print('Evaluation of last pca method')
            distance_models.append(last_pca_component(train_hb))
            models_names.append('Last PCA component')
            distance_functions.append(last_pca_get_distances)
        elif arg == 'first-pca':
            print('Evaluation of first pca method')
            distance_models.append(first_pca_component(train_hb))
            models_names.append('First PCA component')
            distance_functions.append(last_pca_get_distances)
        elif arg == 'pca':
            print('Evaluation of pca method')
            distance_models.append(get_pca_matrix(train_hb))
            models_names.append('PCA k=45')
            distance_functions.append(pca_get_distances)
        elif arg == 'scaler-pca':
            print('Evaluation of scaler-pca method')
            pipeline_model_select = True
        else:
            print('Nothing selected')
            exit(0)
    # evaluation of standard models
    for model, get_distance, model_name in zip(distance_models, distance_functions, models_names):
        new_model_evaluation(model, model_name, get_distance, train_hb, train_cls, test_hb, test_cls)
    # evaluation of scaler pca model
    if pipeline_model_select:
        pipeline_model(train_hb, train_cls, test_hb, test_cls)

