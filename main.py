import numpy as np
import CNN_models
import data_extraction
import result
import utils
import classification
import pickle
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

# # Creation of file raw from file .gdf
# raw_file_train_2a, raw_file_eval_2a = utils.raw_from_gdf(path_2a)
# raw_file_train_2b, raw_file_eval_2b = utils.raw_from_gdf(path_2b)
#
# # Save  variable
# pickle.dump(raw_file_train_2a, open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2a.dat", "wb"))
# pickle.dump(raw_file_train_2b, open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2b.dat", "wb"))
# # Load variable
raw_file_train_2a = pickle.load(open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2a.dat", "rb"))
raw_file_train_2b = pickle.load(open("C:/Users/anto-/PycharmProjects/Project Motor Imagery/Data_Raw/raw_2b.dat", "rb"))

'######################################################################################################################'
''' DATA EXTRACTION + PREPROCESSING '''
# best sub
sub_a = [3, 8]
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
# best_sub
sub_b = [1, 10, 11, 12, 14, 15, 16, 18, 24, 25, 27]
X_b, Y_b = data_extraction.get_data_all_raw(subject=sub_b,
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

'######################################################################################################################'
''' FEATURE EXTRACTION'''
feature = ['hurst_exp', 'svd_fisher_info', 'svd_entropy', 'spect_entropy',
           'teager_kaiser_energy', 'wavelet_coef_energy']

# X ,Y = feature_extraction.extraction_PSD(X, Y)
# X ,Y = feature_extraction.extraction_FE(X, Y, feature)
# X, Y = feature_extraction.extraction_CSP_FE(X, Y, n_component=3, feature=feature)

'######################################################################################################################'
""" CLASSIFICATION """
""" Settings Random Forest Classifier """
step = [
    ('scaler', RobustScaler()),
    ('selector', SelectKBest(mutual_info_classif)),
    ('model', RandomForestClassifier(random_state=42, bootstrap=True))
]
n_estimators = list(np.arange(start=1, stop=150, step=10))
max_depth = list(np.arange(start=1, stop=100, step=10))
min_samples_split = list(np.arange(start=2, stop=10, step=1))
min_samples_leaf = list(np.arange(start=2, stop=10, step=1))
k = list(np.arange(start=10, stop=80, step=5))

param_random_forest = dict(selector_k=k, model__n_estimators=n_estimators,
                           model__max_depth=max_depth,
                           model__min_samples_split=min_samples_split,
                           model__min_samples_leaf=min_samples_leaf
                           )
grid_RF, train_score_RF, test_score_RF, y_test_RF, y_pred_RF = classification.classification(X, Y, step,
                                                                                             param_random_forest, cv=10)
""" Settings SVC Classifier """
step = [
    ('scaler', RobustScaler()),
    ('selector', SelectKBest(mutual_info_classif)),
    ('model', SVC(random_state=42))
]
param_SVC = dict(
    selector__k=k,
    model__C=[0.1, 1, 10, 100, 1000],
    model__gamma=[1, 0.1, 0.01, 0.001, 0.0001],
    model__kernel=['rbf', 'poly', 'linear', 'sigmoid'])

grid_SVC, train_score_SVC, test_score_SVC, y_test_SVC, y_pred_SVC = classification.classification(X, Y, step, param_SVC,
                                                                                                  cv=10)
""" Settings for XGBoost """

n_estimators = list(np.arange(start=1, stop=150, step=10))
max_depth = list(np.arange(start=1, stop=100, step=10))
min_samples_split = list(np.arange(start=2, stop=10, step=1))
min_samples_leaf = list(np.arange(start=2, stop=10, step=1))
gamma = list(np.arange(start=0, stop=1, step=0.25))
learning_rate = list(np.arange(start=0, stop=0.3, step=0.05))
subsample = [0.6, 0.7, 0.8, 0.9]
colsample_by_tree = [0.5, 0.6, 0.7]

step = [
    ('scaler', RobustScaler()),
    ('model', XGBClassifier())
]

param_XGB = dict(
    model__max_depth=max_depth,
    model__learning_rate=learning_rate,
    model__gamma=gamma,
    model__n_estimators=n_estimators,
    model__subsample=subsample,
    model__colsample_bytree=colsample_by_tree
)
grid_XGB, train_score_XGB, test_score_XGB, y_test_XGB, y_pred_XGB = classification.classification(X, Y, step, param_XGB,
                                                                                                  cv=10)

"""Setting TPot Classifier """

TPot = TPOTClassifier(generations=50,
                      population_size=50, subsample=0.9,
                      cv=10,
                      verbosity=2,
                      random_state=42,
                      config_dict="TPOT light",
                      n_jobs=-1)

grid_TP, train_score_TP, test_score_TP, y_test_TP, y_pred_TP = classification.classification_TPOT(X, Y, TPot)

"""Setting CNN Models """

chans, samples = X.shape[1], X.shape[2]

""" EEGNet """
model_EEGNet = CNN_models.EEGNet(nb_classes=2, Chans=chans, Samples=samples,
                                 dropoutRate=0.5, kernLength=64, F1=8,
                                 D=2, F2=16, norm_rate=0.25, dropoutType='Dropout')

model_EEGNet_result, history_EEGNet, scores_EEGNet, y_test_EEGNEt, y_pred_EEGNet = model_classifier(
    "EEGNet", 2,
    X, Y, model_EEGNet,
    batch_size=16,
    epochs=800,
    early_stop=100)

result.result(scores_EEGNet['train'], scores_EEGNet['test'], scores_EEGNet['validation'],
              y_test_EEGNEt, y_pred_EEGNet)

""" DeepConvNet """
model_DeepCovNet = CNN_models.DeepConvNet(nb_classes=2, Chans=chans
                                          , Samples=samples, dropoutRate=0.6)

model_DeepCovNet_result, history_DeepCovNet, scores_DeepCovNet, y_test_DeepCovNet, y_pred_DeepCovNet = model_classifier(
    "DeepCovNet", 2, X, Y, model_DeepCovNet,
    batch_size=16, epochs=200, early_stop=100)

result.result(scores_DeepCovNet['train'], scores_DeepCovNet['test'],
              scores_DeepCovNet['validation'], y_test_DeepCovNet, y_pred_DeepCovNet)

""" ATCNet """
model_ATCNet = CNN_models.ATCNet(n_classes=2, in_chans=chans, in_samples=samples, n_windows=3, attention=None,
                                 eegn_F1=16, eegn_D=2, eegn_kernelSize=128, eegn_poolSize=8, eegn_dropout=0.3,
                                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3,
                                 tcn_activation='elu', fuse='average')

model_ATCNet_result, history_ATCNet, scores_ATCNet, y_test_ATCNet, y_pred_ATCNet = model_classifier(
    "ATCNet", 2, X, Y, model_ATCNet,
    batch_size=16,
    epochs=400, early_stop=100)

result.result(scores_ATCNet['train'], scores_ATCNet['test'],
              scores_ATCNet['validation'], y_test_ATCNet, y_pred_ATCNet)
