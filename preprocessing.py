import mne
from asrpy import ASR
import numpy as np


def set_cap_montage(raw, montage):
    """
    :return: raw with different cap montage
    :param raw:  Raw file
    :param montage:  Path od file .loc montage of raw file
    """
    if raw.filenames[0].__contains__('2a'):
        raw.rename_channels({'EEG-C3': 'C3'})
        raw.rename_channels({'EEG-Cz': 'Cz'})
        raw.rename_channels({'EEG-C4': 'C4'})
        raw.rename_channels({'EOG-left': 'EOG_0'})
        raw.rename_channels({'EOG-central': 'EOG_1'})
        raw.rename_channels({'EOG-right': 'EOG_2'})
    elif raw.filenames[0].__contains__('2b'):
        raw.rename_channels({'EEG:C3': 'C3'})
        raw.rename_channels({'EEG:Cz': 'Cz'})
        raw.rename_channels({'EEG:C4': 'C4'})
        raw.rename_channels({'EOG:ch01': 'EOG_0'})
        raw.rename_channels({'EOG:ch02': 'EOG_1'})
        raw.rename_channels({'EOG:ch03': 'EOG_2'})

    montage_var = mne.channels.read_custom_montage(montage)
    raw.set_montage(montage_var)

    return raw


def raw_filter(raw, f_min, f_max):
    """
    :param raw: file raw
    :param f_min: min frequency of filter
    :param f_max: max frequnct of filter
    :return: raw filtred
    """
    raw.filter(l_freq=f_min, h_freq=f_max, fir_design='firwin')
    return raw


def raw_drop_channels(raw, channels):
    """
    :param raw: file raw
    :param channels: list of channel name to drop
    :return: raw dropped of channels
    """
    raw.drop_channels(channels)
    return raw


def artifact_correction_ASR(raw, cutoff):
    """
    Artifact Subspace Reconstruction
    :param raw: file raw
    :param cutoff: Standard deviation cutoff for rejection. X portions whose variance
        is larger than this threshold relative to the calibration data are
        considered missing data and will be removed.
        Reccomended from 20 - 30
    :return:file raw processed
    todo: filter the file raw before doing artifact correction
          set filter from 1 to None
    """
    asr = ASR(raw.info["sfreq"], cutoff=cutoff)
    asr.fit(raw)
    raw = asr.transform(raw)
    return raw


def artifact_correction_ICA(raw, n_component):
    """
        Indipendent Component Analysis
        :param raw: file raw
        :param n_component:
        :return: file raw processed
        todo: filter the file raw before doing artifact correction
              set filter from 1 to None
        """
    ica = mne.preprocessing.ICA(n_components=n_component, random_state=42, method="infomax")
    ica.fit(raw)
    eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw, ch_name=['EOG_0', 'EOG_1', 'EOG_2'])
    # Find bads eog channels from find_bads_eog
    threshold = 0.5
    eog_inds, scores = ica.find_bads_eog(eog_epochs, ch_name='EOG_0')

    # Apply ICA
    ica.exclude = np.argwhere(abs(scores) > threshold).ravel().tolist()
    raw = ica.apply(raw)
    return raw


def raw_preprocessing(raw, montage, fmin, fmax, channels):
    """
    Simple Preprocessing filter + drop channel
    :param raw: file raw
    :param montage:  Path od file .loc montage of raw file
    :param fmin: min frequency of filter
    :param fmax: max frequency of filter
    :param channels: list of channel name to drop
    :return: raw processed
    """
    raw = set_cap_montage(raw, montage)
    raw = raw_drop_channels(raw, channels)
    raw = raw_filter(raw, fmin, fmax)

    return raw


def raw_preprocessing_ASR(raw, montage, f_min, f_max, channels, cutoff):
    """
    Raw preprocessing through Artifact Subspace Recostruction
    :param cutoff: int
    :param raw: file raw
    :param montage:  Path od file .loc montage of raw file
    :param f_min: min frequency of filter
    :param f_max: max frequency of filter
    :param channels: list of channel name to drop
    :return: raw processed
    """
    if cutoff is not None:
        raw = set_cap_montage(raw, montage)
        raw = raw_filter(raw, 1, None)
        raw = artifact_correction_ASR(raw, cutoff)
        raw = raw_drop_channels(raw, channels)
        raw = raw_filter(raw, f_min, f_max)
    else:
        raise Exception("Error: cutoff must not be None")
    return raw


def raw_preprocessing_ICA(raw, montage, f_min, f_max, channels, n_component):
    """
    Raw preprocessing through Indipendet Component Analysis
    :param n_component:
    :param raw: file raw
    :param montage:  Path od file .loc montage of raw file
    :param f_min: min frequency of filter
    :param f_max: max frequency of filter
    :param channels: list of channel name to drop
    :return: raw processed
    """
    if n_component is not None:
        raw = set_cap_montage(raw, montage)
        raw = raw_filter(raw, 1, None)
        raw = artifact_correction_ICA(raw, n_component=n_component)
        raw = raw_drop_channels(raw, channels)
        raw = raw_filter(raw, f_min, f_max)
    else:
        raise Exception("Error: n_component must not be None")
    return raw
