import os
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
    print(len(train_hb), len(test_cls))
    pca_matrix = get_pca_matrix(train_hb)
    distance_models = [pca_matrix]
    models_names = ['PCA k=45']
    distance_functions = [pca_get_distances]
    for model, get_distance, model_name in zip(distance_models, distance_functions, models_names):
        new_model_evaluation(model, model_name, get_distance, train_hb, train_cls, test_hb, test_cls)
    pipeline_model(train_hb, train_cls, test_hb, test_cls)
