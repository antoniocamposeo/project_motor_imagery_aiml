""" TEST 5 -  classification on single subjects with 2comp csp and 4comp csp and with RadomForest e TPot"""
import numpy as np
import data_extraction
import classification
import pickle
import feature_extraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from tpot import TPOTClassifier


# PATH OF FILE GDF AND CAP
path_2a = 'C:/Users/anto-/PycharmProjects/AIML_Project/BCICIV_2a_gdf'
montage_path_2a = 'C:/Users/anto-/PycharmProjects/AIML_Project/Montages/montage_2a.loc'

path_2b = 'C:/Users/anto-/PycharmProjects/AIML_Project/BCICIV_2b_gdf'
montage_path_2b = 'C:/Users/anto-/PycharmProjects/AIML_Project/Montages/montage_2b.loc'

# Channels to drop out from raw files 2a and 2b
channels_drop_2a = ['EOG_1', 'EOG_0', 'EOG_2', 'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2'
    , 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-6', 'EEG-7', 'EEG-8'
    , 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-15',
                    'EEG-16', 'EEG-Pz']
channels_drop_2b = ['EOG_1', 'EOG_0', 'EOG_2']

test = {'1': {'rf',2}, '2': {'rf',3}, '3': {'tpot',2}, '4': {'tpot',3}}
dict_a = {}
for k in test.keys():
    raw_file_train_2a = pickle.load(
        open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2a.dat", "rb"))
    value = test[f"{k}"]
    for i in range(1, 10):
        sub_a = [i]
        X_a, Y_a = data_extraction.get_data_all_raw(
            subject=sub_a,
            raw_files=raw_file_train_2a,
            montage=montage_path_2a,
            f_min=8,
            f_max=30,
            t_min=0,
            t_max=3,
            channel_drop=channels_drop_2a,
            class_event=['left_hand', 'right_hand'],
            artifact_correction=None)

        X, Y = feature_extraction.extraction_CSP(X_a, Y_a, test[f"{k}"][1])
        if test[f"{k}"][0] == 'rf':
            step = [
                ('scaler', RobustScaler()),
                ('model', RandomForestClassifier(random_state=42, bootstrap=True))
            ]

            n_estimators = list(np.arange(start=1, stop=50, step=20))
            max_depth = list(np.arange(start=1, stop=40, step=15))
            min_samples_split = list(np.arange(start=2, stop=8, step=2))
            min_samples_leaf = list(np.arange(start=2, stop=8, step=2))

            param_random_forest = dict(model__n_estimators=n_estimators,
                                       model__max_depth=max_depth,
                                       model__min_samples_split=min_samples_split,
                                       model__min_samples_leaf=min_samples_leaf
                                       )
            grid_RF, train_score_RF, test_score_RF, y_test_RF, y_pred_RF = classification.classification(X, Y, step,
                                                                                                         param_random_forest,
                                                                                                         cv=10)
            score_train = train_score_RF
            score_test = test_score_RF

        elif test[f"{k}"][0] == 'tpot':

            TPot = TPOTClassifier(generations=50,
                                  population_size=50, subsample=0.7,
                                  cv=10,
                                  verbosity=2,
                                  random_state=42,
                                  config_dict="TPOT light",
                                  n_jobs=-1)

            grid_TP, train_score_TP, test_score_TP, y_test_TP, y_pred_TP = classification.classification_TPOT(X,
                                                                                                              Y,
                                                                                                              TPot)
            score_train = train_score_TP
            score_test = test_score_TP
        else:
            raise Exception()

        dict_a[f"{value}+2a:s{i}"] = (score_train,score_test)

dict_b = {}
for k in test.keys():
    raw_file_train_2b = pickle.load(
        open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2b.dat", "rb"))
    value = test[f"{k}"]
    for i in range(1, 10):
        sub_b = [i]
        X_b, Y_b = data_extraction.get_data_all_raw(
            subject=sub_b,
            raw_files=raw_file_train_2b,
            montage=montage_path_2b,
            f_min=8,
            f_max=30,
            t_min=1,
            t_max=4,
            channel_drop=channels_drop_2b,
            class_event=['left_hand', 'right_hand'],
            artifact_correction=None)

        X, Y = feature_extraction.extraction_CSP(X_b, Y_b, test[f"{k}"][1])
        if test[f"{k}"][0] == 'rf':
            step = [
                ('scaler', RobustScaler()),
                ('model', RandomForestClassifier(random_state=42, bootstrap=True))
            ]

            n_estimators = list(np.arange(start=1, stop=50, step=20))
            max_depth = list(np.arange(start=1, stop=40, step=15))
            min_samples_split = list(np.arange(start=2, stop=8, step=2))
            min_samples_leaf = list(np.arange(start=2, stop=8, step=2))

            param_random_forest = dict(model__n_estimators=n_estimators,
                                       model__max_depth=max_depth,
                                       model__min_samples_split=min_samples_split,
                                       model__min_samples_leaf=min_samples_leaf
                                       )
            grid_RF, train_score_RF, test_score_RF, y_test_RF, y_pred_RF = classification.classification(X, Y, step,
                                                                                                         param_random_forest,
                                                                                                         cv=10)
            score_train = train_score_RF
            score_test = test_score_RF

        elif test[f"{k}"][0] == 'tpot':

            TPot = TPOTClassifier(generations=50,
                                  population_size=50, subsample=0.7,
                                  cv=10,
                                  verbosity=2,
                                  random_state=42,
                                  config_dict="TPOT light",
                                  n_jobs=-1)

            grid_TP, train_score_TP, test_score_TP, y_test_TP, y_pred_TP = classification.classification_TPOT(X,
                                                                                                              Y,
                                                                                                              TPot)
            score_train = train_score_TP
            score_test = test_score_TP
        else:
            raise Exception()

        dict_b[f"{value}2b:s{i}"] = (score_train,score_test)

import pandas as pd
df_a = pd.DataFrame.from_dict(dict_a)
df_b = pd.DataFrame.from_dict(dict_b)
df_a.to_excel('C:/Users/anto-/PycharmProjects/AIML_Project/File Excel/test_5_2a.xlsx')
df_b.to_excel('C:/Users/anto-/PycharmProjects/AIML_Project/File Excel/test_5_2b.xlsx')

