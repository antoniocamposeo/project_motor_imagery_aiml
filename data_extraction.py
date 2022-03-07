import numpy as np
import preprocessing
import mne
import utils


def data_extract_epochs(raw, t_min, t_max, class_event):
    """
    :param raw: array raw file
    :param t_min: float epoch min time
    :param t_max: float epoch max time
    :param class_event: array of class type ['left_hand','right_hand','rest']
    :return: data of epochs and labels refers to epochs
    """
    events, event_dict = mne.events_from_annotations(raw)
    if class_event.__contains__('left_hand') and class_event.__contains__('right_hand') \
            and not class_event.__contains__('rest'):
        event_dict = {'769': event_dict['769'], '770': event_dict['770']}
        epochs = mne.Epochs(raw, events, event_dict, t_min, t_max, proj=True,
                            baseline=None, preload=True, verbose=True)
        data_epochs = epochs.get_data()
        data_labels = epochs.events[:, -1] - event_dict['769']

    elif class_event.__contains__('left_hand') and class_event.__contains__('rest') \
            and not class_event.__contains__('right_hand'):
        event_dict = {'768': event_dict['768'], '769': event_dict['769']}
        epochs = mne.Epochs(raw, events, event_dict, t_min, t_max, proj=True,
                            baseline=None, preload=True, verbose=True)
        data_epochs = epochs.get_data()
        data_labels = epochs.events[:, -1] - event_dict['768']
        data_epochs, data_labels = utils.reduce_epochs_2a(raw, data_epochs, data_labels)
    elif class_event.__contains__('right_hand') and class_event.__contains__('rest') \
            and not class_event.__contains__('left_hand'):
        event_dict = {'768': event_dict['768'], '770': event_dict['770']}
        epochs = mne.Epochs(raw, events, event_dict, t_min, t_max, proj=True,
                            baseline=None, preload=True, verbose=True)
        data_epochs = epochs.get_data()
        data_labels = epochs.events[:, -1] - event_dict['768']
        data_epochs, data_labels = utils.reduce_epochs_2a(raw, data_epochs, data_labels)
    elif class_event.__contains__('right_hand') and class_event.__contains__('rest') \
            and class_event.__contains__('rest'):
        event_dict = {'768': event_dict['768'], '769': event_dict['769'], '770': event_dict['770']}
        epochs = mne.Epochs(raw, events, event_dict, t_min, t_max, proj=True,
                            baseline=None, preload=True, verbose=True)
        data_epochs = epochs.get_data()
        data_labels = epochs.events[:, -1] - event_dict['768']
        data_epochs, data_labels = utils.reduce_epochs_2a(raw, data_epochs, data_labels)
    elif class_event is None:
        raise Exception("Error: class_event is empty")
    else:
        raise Exception("Error: String in event")

    return data_epochs, data_labels


def get_data_raw(raw, montage, f_min, f_max, t_min, t_max, channel_drop, class_event, artifact_correction):
    """
    :param raw: array raw file
    :param montage: String
    :param f_min: float
    :param f_max: float
    :param t_min: float epoch min time
    :param t_max:  float epoch max time
    :param channel_drop: Array of channels to drop
    :param class_event: Array of class event -- ['left_hand','right_hand','rest']
    :return: data of epochs and lables refers to epoch
    """
    if artifact_correction is None:
        raw = preprocessing.raw_preprocessing(raw, montage, f_min, f_max, channel_drop)
    elif artifact_correction == 'ASR':
        raw = preprocessing.raw_preprocessing_ASR(raw, montage, f_min, f_max, channel_drop, 20)
    elif artifact_correction == 'ICA':
        raw = preprocessing.raw_preprocessing_ICA(raw, montage, f_min, f_max, channel_drop, 4)
    else:
        raise
    data_epochs, data_labels = data_extract_epochs(raw, t_min, t_max, class_event)
    return data_epochs, data_labels


def get_data_all_raw(subject, raw_files, montage, f_min, f_max, t_min, t_max, channel_drop, class_event,artifact_correction):
    """
    :param subject: array of best subject
    :param raw_files: array of files raw
    :param montage: cap of data
    :param f_min: float
    :param f_max: float
    :param t_min: float
    :param t_max: float
    :param channel_drop: array
    :param class_event: array
    :return:
    """
    X = []
    Y = []
    if subject is None:
        for i in range(0, len(raw_files)):
            raw = raw_files[i]
            data_epochs, data_labels = get_data_raw(raw, montage,
                                                    f_min=f_min,
                                                    f_max=f_max,
                                                    t_min=t_min, t_max=t_max,
                                                    channel_drop=channel_drop,
                                                    class_event=class_event,artifact_correction=artifact_correction)

            X.append(data_epochs)
            Y.append(data_labels)
    elif subject is not None:
        for i in subject:
            raw = raw_files[i-1]
            data_epochs, data_labels = get_data_raw(raw, montage,
                                                    f_min=f_min,
                                                    f_max=f_max,
                                                    t_min=t_min, t_max=t_max,
                                                    channel_drop=channel_drop,
                                                    class_event=class_event,artifact_correction=artifact_correction)

            X.append(data_epochs)
            Y.append(data_labels)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X, Y
