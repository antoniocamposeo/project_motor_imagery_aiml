""" TEST 3"""
import numpy as np
import classification
import data_extraction
import pickle
from tpot import TPOTClassifier

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

all_feature = ['energy_freq_bands','hjorth_complexity', 'hjorth_complexity_spect',
               'hjorth_mobility', 'hjorth_mobility_spect',
               'hurst_exp',  'kurtosis',
               'line_length', 'mean', 'pow_freq_bands',
               'ptp_amp', 'quantile', 'rms', 'samp_entropy',
               'skewness', 'spect_edge_freq', 'spect_entropy',
               'spect_slope', 'std', 'svd_entropy', 'svd_fisher_info',
               'teager_kaiser_energy', 'variance', 'wavelet_coef_energy',
               'zero_crossings', 'max_cross_corr',
               'nonlin_interdep', 'phase_lock_val', 'spect_corr', 'time_corr']

index = []
for i in range(1, len(all_feature)+1):
    index.append(np.arange(0, i, 1))
feature_array = []
for k in index:
    temp = []
    for value in k:
        temp.append(all_feature[value])
    feature_array.append(temp)

raw_file_train_2a = pickle.load(open("C:/Users/anto-/PycharmProjects/AIML_Project/Data_Raw/raw_2a.dat", "rb"))
raw_file_train_2b = pickle.load(open("C:/Users/anto-/PycharmProjects/AIML_Project/Data_Raw/raw_2b.dat", "rb"))

'######################################################################################################################'
''' DATA EXTRACTION + PREPROCESSING '''
# best sub
X_a, Y_a = data_extraction.get_data_all_raw(
    subject=None,
    raw_files=raw_file_train_2a,
    montage=montage_path_2a,
    f_min=8,
    f_max=30,
    t_min=0,
    t_max=3,
    channel_drop=channels_drop_2a,
    class_event=['left_hand', 'right_hand'],
    artifact_correction=None)
# best_sub
X_b, Y_b = data_extraction.get_data_all_raw(subject=None,
                                            raw_files=raw_file_train_2b,
                                            montage=montage_path_2b,
                                            f_min=8,
                                            f_max=30,
                                            t_min=1,
                                            t_max=4,
                                            channel_drop=channels_drop_2b,
                                            class_event=['left_hand', 'right_hand'],
                                            artifact_correction=None)
# Concat of two different dataset 2a+2b
X = np.concatenate((X_a, X_b), axis=0)
Y = np.concatenate((Y_a, Y_b), axis=0)
score_train = []
score_test = []

import feature_extraction

result_feature = {}
for feature in feature_array:
    X_temp, Y_temp = feature_extraction.extraction_CSP_FE(X, Y, 2, feature)

    TPot = TPOTClassifier(generations=50,
                          population_size=50, subsample=0.7,
                          cv=10,
                          verbosity=2,
                          random_state=42,
                          config_dict="TPOT light",
                          n_jobs=-1)

    grid_TP, train_score_TP, test_score_TP, y_test_TP, y_pred_TP = classification.classification_TPOT(X_temp, Y_temp, TPot)

    result_feature[feature.__str__()] = (train_score_TP, test_score_TP)


import pandas as pd
df = pd.DataFrame.from_dict(result_feature)
df.to_excel('../File Excel/test_3.xlsx')