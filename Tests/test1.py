import numpy as np
import CNN_models
import data_extraction
import feature_extraction
import result
import utils
import classification
import pickle
import feature_extraction

from CNN_classification import model_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import RobustScaler
from tpot import TPOTClassifier
from xgboost import XGBClassifier

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

raw_file_train_2a = pickle.load(open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2a.dat", "rb"))
raw_file_train_2b = pickle.load(open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2b.dat", "rb"))
score_test = []
score_train = []

for i in range(1, 28):
    sub_a = [i]
    X_a, Y_a = data_extraction.get_data_all_raw(
        subject=sub_a,
        raw_files=raw_file_train_2a,
        montage=montage_path_2a,
        f_min=8,
        f_max=30,
        t_min=1,
        t_max=4,
        channel_drop=channels_drop_2a,
        class_event=['left_hand', 'right_hand'],
        artifact_correction=None)


    X, Y = feature_extraction.extraction_PSD(X_a, Y_a)

    step = [
        ('scaler', RobustScaler()),
        ('model', RandomForestClassifier(random_state=42, bootstrap=True))
    ]

    n_estimators = list(np.arange(start=1, stop=150, step=30))
    max_depth = list(np.arange(start=1, stop=100, step=20))
    min_samples_split = list(np.arange(start=2, stop=10, step=2))
    min_samples_leaf = list(np.arange(start=2, stop=10, step=2))

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

for i in range(1, 28):
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

    import feature_extraction

    X, Y = feature_extraction.extraction_PSD(X_b,Y_b)

    step = [
        ('scaler', RobustScaler()),
        ('model', RandomForestClassifier(random_state=42, bootstrap=True))
    ]

    n_estimators = list(np.arange(start=1, stop=150, step=30))
    max_depth = list(np.arange(start=1, stop=100, step=20))
    min_samples_split = list(np.arange(start=2, stop=10, step=2))
    min_samples_leaf = list(np.arange(start=2, stop=10, step=2))

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