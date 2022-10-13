""" TEST 4 """
import classification
import data_extraction
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler


# PATH OF FILE GDF AND CAP
path_2a = 'C:/Users/anto-/PycharmProjects/Project Motor Imagery/BCICIV_2a_gdf'
montage_path_2a = 'C:/Users/anto-/PycharmProjects/Project Motor Imagery/Montages/montage_2a.loc'

path_2b = 'C:/Users/anto-/PycharmProjects/Project Motor Imagery/BCICIV_2b_gdf'
montage_path_2b = 'C:/Users/anto-/PycharmProjects/Project Motor Imagery/Montages/montage_2b.loc'

# Channels to drop out from raw files 2a and 2b
channels_drop_2a = ['EOG_1', 'EOG_0', 'EOG_2', 'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2'
    , 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-6', 'EEG-7', 'EEG-8'
    , 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-15',
                    'EEG-16', 'EEG-Pz']
channels_drop_2b = ['EOG_1', 'EOG_0', 'EOG_2']

# creazione gruppi
"""
[1 2 3 4 5 6 7 8 ]
[[1 2 3],[2 3 4],[3 4 5]----- [6 7 8] [8 1 2] [1 2 3]]

[[1 2] [2 3] [3 4] [4 5]

for i in range(1,len(group_2a)-1):
    arr_2a.append([group_2a[i],group_2a[i+1])
    
    
"""
import numpy as np
group_2b = np.arange(0,27,1)
arr_2b = []
step_max = 2
for i in range(1,26):
    if i > 1:
        arr_2b.append([group_2b[i-step_max+1], group_2b[i], group_2b[i+step_max-1]])
    elif i == len(group_2b):
        arr_2b.append([group_2b[i-step_max], group_2b[i-step_max+1], group_2b[i]])

group_2a = np.arange(0,10,1)
arr_2a = []
step_max = 2
for i in range(1,9):
    if i > 1:
        arr_2a.append([group_2a[i-step_max+1], group_2a[i], group_2a[i+step_max-1]])
    elif i == len(group_2a):
        arr_2a.append([group_2a[i-step_max], group_2a[i-step_max+1], group_2a[i]])

import feature_extraction
group_values_a = {}

for k in arr_2a:
    raw_file_train_2a = pickle.load(
        open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2a.dat", "rb"))
    score_train = []
    score_test = []
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

        X,Y = feature_extraction.extraction_CSP(X_a,Y_a,2)

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
        score_train.append(train_score_RF)
        score_test.append(test_score_RF)

    group_values_a[f"{k}"] = (np.mean(score_train),np.mean(score_test))


# 2B
group_values_b = {}

for k in arr_2b:
    raw_file_train_2b = pickle.load(
        open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2a.dat", "rb"))
    score_train = []
    score_test = []
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

        X, Y = feature_extraction.extraction_CSP(X_b, Y_b, 2)

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
        score_train.append(train_score_RF)
        score_test.append(test_score_RF)

    group_values_a[f"{k}"] = (np.mean(score_train), np.mean(score_test))

import pandas as pd
df = pd.DataFrame.from_dict(group_values_b)
df.to_excel('C:/Users/anto-/PycharmProjects/AIML_Project/File Excel/test_4_2b.xlsx')

import pandas as pd
df = pd.DataFrame.from_dict(group_values_a)
df.to_excel('C:/Users/anto-/PycharmProjects/AIML_Project/File Excel/test_4_2a.xlsx')




