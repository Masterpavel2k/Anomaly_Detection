import os
from avgHbModel import average_heart_beat, get_distances as avg_get_distances
from convertEcgInCsv import convert_ecg_in_csv
from datasetExtraction import get_sets
from distanceModelEvaluation import new_model_evaluation
from downloadEcg import download_ecg
from pcaModel import get_pca_matrix, get_distances as pca_get_distances
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
    size = 1000
    # download, preprocess and convert ecg if necessary
    download_ecg(cwd, normal_folder_name, abnormal_folder_name)
    new_normal_folder, new_abnormal_folder = preprocess_ecg(cwd, normal_folder_name, abnormal_folder_name)
    convert_ecg_in_csv(cwd, new_normal_folder, new_abnormal_folder, normal_file_name, abnormal_file_name)
    # start models comparison
    train_hb, train_cls, test_hb, test_cls = get_sets(size, cwd, normal_file_name, abnormal_file_name)
    distance_models = [average_heart_beat(train_hb), get_pca_matrix(train_hb),
                       first_pca_component(train_hb), last_pca_component(train_hb)]
    distance_functions = [avg_get_distances, pca_get_distances, last_pca_get_distances, last_pca_get_distances]
    for model, get_distance in zip(distance_models, distance_functions):
        new_model_evaluation(model, get_distance, train_hb, train_cls, test_hb, test_cls)
    pipeline_model(train_hb, train_cls, test_hb, test_cls)
