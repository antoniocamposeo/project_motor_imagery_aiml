from os import listdir
from os.path import isfile, join
import mne
import numpy as np


def raw_from_gdf(path):
    """
    :param: path of gdf files
    :return: vector of train raw files and eval/test raw files

    """

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # list of directory
    mypath_list_train = []
    mypath_list_eval = []

    for i in range(0, len(onlyfiles)):
        if "T" in onlyfiles[i]:
            mypath_list_train.append(path + '/' + onlyfiles[i])
        else:
            mypath_list_eval.append(path + '/' + onlyfiles[i])

    # Vector of file raw
    raw_files_train = [
        mne.io.read_raw_gdf(mypath_list_train[i], preload=True) for i in range(0, len(mypath_list_train))
    ]
    # Vector of file raw
    raw_files_eval = [
        mne.io.read_raw_gdf(mypath_list_eval[i], preload=True) for i in range(0, len(mypath_list_eval))
    ]
    return raw_files_train, raw_files_eval


def reduce_epochs_2a(raw, epochs_data, epochs_labels):
    """
    implementation of this function in order to reduce the class rest into
    raw 2a files
    :param raw:
    :param epochs_data: data from epoch  array(,,,)
    :param epochs_labels: labels from epoch array(,1)
    :return:
    """
    if raw.filenames[0].__contains__('2a'):
        res = np.where(epochs_data == 0)[0]
        temp = []
        temp_1 = []
        for i in range(0, len(res)):
            if i < round(len(res) / 2):
                temp.append(res[i])
            else:
                temp_1.append(res[i])
        data_x = np.delete(epochs_data, temp_1, axis=0)
        data_y = np.delete(epochs_labels, temp_1, axis=0)
        return data_x, data_y
    else:
        return epochs_data, epochs_labels

