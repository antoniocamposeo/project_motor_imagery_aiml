import numpy as np
from mne.decoding import CSP
from mne_features.feature_extraction import FeatureExtractor
import neurokit2 as nk

all_feature = ['app_entropy', 'decorr_time',
               'energy_freq_bands', 'higuchi_fd',
               'hjorth_complexity', 'hjorth_complexity_spect',
               'hjorth_mobility', 'hjorth_mobility_spect',
               'hurst_exp', 'katz_fd', 'kurtosis',
               'line_length', 'mean', 'pow_freq_bands',
               'ptp_amp', 'quantile', 'rms', 'samp_entropy',
               'skewness', 'spect_edge_freq', 'spect_entropy',
               'spect_slope', 'std', 'svd_entropy', 'svd_fisher_info',
               'teager_kaiser_energy', 'variance', 'wavelet_coef_energy',
               'zero_crossings', 'max_cross_corr',
               'nonlin_interdep', 'phase_lock_val', 'spect_corr', 'time_corr']


def get_csp(n_component):
    """
    Common Spatial Pattern from MNE Decoding
    :param n_component: int
        The number of components to decompose M/EEG signals.
    :return: csp
    """
    csp = CSP(n_components=n_component, reg=None, log=True, norm_trace=False)
    return csp


def get_feature_extractor(feature):
    """
    Object Feature Extraction from mne_features
    :param feature:String Array
        array of feature string contained into all_feature
    :return: fe object
    """
    if set(feature).issubset(set(all_feature)):
        fe = FeatureExtractor(250, selected_funcs=feature)
    else:
        raise Exception('Error: features are not correct')
    return fe


def extraction_CSP(epochs_data, epochs_lables, n_component):
    """
    Common Spatial Pattern to retrieve the component signals
    :param epochs_data: Array(,,,)
        param create from data_extraction
    :param epochs_lables: Array(,1)
        param create from data_extraction
    :param n_component: int
        The number of components to decompose M/EEG signals.
    :return: Array(n_epochs,n_component),Array(1,n_labels )
    """
    csp = get_csp(n_component=n_component)
    X = csp.fit_transform(epochs_data, epochs_lables)
    Y = epochs_lables
    return X, Y


def extraction_FE(epochs_data, epochs_lables, feature):
    """
    Extraxtion Features
    :param feature:
    :param epochs_data: Array(,,,)
        param create from data_extraction
    :param epochs_lables: Array(,1)
        param create from data_extraction
    :return: Array(N_EPOCHS,N_FEATURES),Array(1,n_labels )
    """
    fe = get_feature_extractor(feature=feature)
    X = fe.fit_transform(epochs_data, epochs_lables)
    Y = epochs_lables
    return X, Y


def extraction_CSP_FE(epochs_data, epochs_lables, n_component, feature):
    """
    Concatenate CSP + feature extraction
    :param epochs_data: Array(,,,)
        param create from data_extraction
    :param epochs_lables: Array(,1)
        param create from data_extraction
    :param n_component: int
        The number of components to decompose M/EEG signals.
    :param feature : String array
        array of feature string contained into all_feature
    :return: Array(n_epochs,n_feature+n_component),Array(1,n_labels )
    """
    csp = get_csp(n_component=n_component)
    fe = get_feature_extractor(feature=feature)
    X_csp = csp.fit_transform(epochs_data, epochs_lables)
    X_fe = fe.fit_transform(epochs_data, epochs_lables)
    X = np.concatenate((X_csp, X_fe), axis=1)
    Y = epochs_lables

    return X, Y


def extraction_PSD(data_epochs, data_labels):
    """
    Power Spectral Density of each band of eeg signal
    :param data_epochs: data extract from epochs
    :param data_labels: labels
    :return: X,Y data extraction
    """
    X = []
    Y = []
    for i in range(0, data_epochs.shape[0]):
        psd_band = nk.eeg_power(data_epochs[i], 250,
                                frequency_band=['Alpha', 'Beta1', 'Beta2', 'Beta3',
                                                'Mu']).to_numpy()
        psd_band = np.delete(psd_band, 0, 1)
        X.append(np.array(psd_band).flatten())
        Y.append(data_labels[i])

    X = np.stack(X, axis=0)
    Y = np.array(Y)
    X = X.astype(float)
    return X, Y
