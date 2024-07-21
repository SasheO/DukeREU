# file_loader.py
# Author: Kevin Chu
# Last Modified: 10/04/2020

# External
import numpy as np
import torch

# Internal
from phone_mapping import map_phones


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, conf_dict, le=None):
        """
        Args:
            file_list (list): list of utterance files
            conf_dict (dict): dictionary with configuration parameters
            le (sklearn.LabelEncoder): used to encode strings as ints
        """
        self.file_list = file_list
        self.conf_dict = conf_dict
        self.le = le

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # Read single feature file
        X, y, phones = read_feat_file(self.file_list[index], self.conf_dict)

        # Only include feature files where forced alignment was successful
        # (i.e. only include files where not all the phones are h#)
        if not np.all(phones == 'h#'):
            # Map phones to categories and encode as integers
            if bool(self.le):
                phones = map_phones(list(phones), self.conf_dict)
                phones = self.le.transform(phones).astype('long')

            # Create chunks
            X = create_chunks(X, self.conf_dict)
            y = create_chunks(y, self.conf_dict)
            phones = create_chunks_phones(phones, self.conf_dict)

        # Exclude feature files where forced alignment was not successful
        else:
            print("File {} is missing forced alignments. Excluding.".format(self.file_list[index]))
            X = np.empty(shape=(0, self.conf_dict["chunk_len"], np.shape(X)[1]), dtype=np.float32)
            y = np.empty(shape=(0, self.conf_dict["chunk_len"], np.shape(y)[1]), dtype=np.float32)
            phones = np.empty(shape=(0, self.conf_dict["chunk_len"]), dtype=np.float32)

        return X, y, phones


def collate_fn(batch):
    """
    Used to collate multiple files for dataloader
    """
    X_batch, y_batch, p_batch = zip(*batch)
    X_batch = np.vstack(list(X_batch))
    y_batch = np.vstack(list(y_batch))
    p_batch = np.vstack(list(p_batch))

    return X_batch, y_batch, p_batch


def read_feat_list(feat_list_file):
    """
    This function reads in the file with the list of feature files.

    Args:
        feat_list_file (str): file with list of feature files

    Returns:
        none
    """
    # Read in file list
    file_obj = open(feat_list_file, "r")
    file_list = file_obj.readlines()
    file_obj.close()
    
    # Remove newline characters
    file_list = list(map(lambda x: x.replace("\n",""), file_list))
    
    return file_list


def read_feat_file(filename, conf_dict):
    """
    This function reads features and labels from a text file.

    Args:
        filename (str): name of text file containing features and labels
        label_type (str): label type

    Returns:
        X (np.array): matrix of features
        y (list): list of phoneme labels
    """
    # If disconnected from server, keep trying to read from file
    io_flag = True
    while io_flag:
        filename = filename.replace(".txt", ".npz")
        try:
            data = np.load(filename)
            io_flag = False
        except OSError:
            print("File not found: " + filename + " . Attempting to read again.")
            continue

    X = data["feats"]
    y = data["mask"]
    phones = data["phones"]

    # Deltas
    X = calculate_deltas(X, conf_dict)

    # Splice
    X = splice(X, conf_dict)

    # Clip mask values
    if "clip_mask" in conf_dict.keys():
        y[y > conf_dict["clip_mask"]] = conf_dict["clip_mask"]

    return X, y, phones


def remove_corpus_channel(filename, conf_dict, X):
    """ 
    This function removes the corpus channel from the features.
    The corpus channel is defined as the stationary component of
    the log magnitude spectrum and is composed of the long term
    spectrum of speech and microphone frequency response. The
    differences between speech corpora are primarily due to
    differences in recording conditions.

    Args:
        filename (str): feature file
        conf_dict (dict): dictionary with configuraiton parameters
        X (np.array): features

    Returns:
        X (np.array): feature matrix with corpus channel removed
    """
    
    if "hint" in filename:
        corpus = "hint"
    else:
        corpus = "timit"

    corpus_channel_file = "corpus_channel/" + corpus + "_channel.txt"
    corpus_channel = np.loadtxt(corpus_channel_file)
    corpus_channel = corpus_channel.astype("float32")
    corpus_channel = np.reshape(corpus_channel, (1, np.shape(corpus_channel)[0]))
    X = X - 2*corpus_channel

    return X


def calculate_deltas(X, conf_dict):
    """ 
    This function calculates first and second order derivatives
    of the features (i.e. deltas and delta-deltas).

    Args:
        X (np.array): matrix of static features
        conf_dict (dict): dictionary with configuration parameters

    Returns:
        X (np.array): feature matrix with deltas and delta-deltas
        concatenated, where applicable
    """
    
    # Deltas and delta-deltas calculated causally
    if conf_dict["deltas"]:
        deltas = np.concatenate((np.zeros((1, np.shape(X)[1]), dtype='float32'), np.diff(X, axis=0)), axis=0)
        X = np.concatenate((X, deltas), axis=1)
        if conf_dict["deltaDeltas"]:
            delta_deltas = np.concatenate(
                (np.zeros((1, np.shape(deltas)[1]), dtype='float32'), np.diff(deltas, axis=0)), axis=0)
            X = np.concatenate((X, delta_deltas), axis=1)

    return X


def splice(X, conf_dict):
    """
    This function concatenates a feature matrix with features
    from causal time frames. The purpose of this is to increase
    the amount of temporal context available to the model.

    Args:
        X (np.array): matrix of features from current time frame
        conf_dict (dict): dictionary with configuration parameters

    Returns:
        X (np.array): feature matrix with causal time frames
        concatenated
    """

    if "window_size" in conf_dict.keys():
        if conf_dict["window_size"] > 1:
            x0 = np.zeros((conf_dict["window_size"]-1, np.shape(X)[1]), dtype='float32')
            X = np.concatenate((x0, X), axis=0)

            # Splice
            batch_sz = np.shape(X)[0] - conf_dict["window_size"] + 1
            idx = np.linspace(0, conf_dict["window_size"]-1, conf_dict["window_size"])
            idx = np.tile(idx, (batch_sz, 1)) + np.linspace(0, batch_sz-1, batch_sz).reshape((batch_sz, 1))
            idx = idx.astype(int)
            X = X[idx, :]
            X = X.reshape(np.shape(X)[0], conf_dict["num_features"])

    return X


def create_chunks(X, conf_dict):
    """
    This function splits utterances into chunks. The purpose
    of this is two-fold: to overcome the problem of training
    RNNs using long sequences and problems related to memory.

    Args:
        X (np.array): (seq_len x n_feats) matrix of features
        (or masks) across time steps
        conf_dict (dict): dictionary with configuration parameters

    Returns:
        X (np.array): (n_chunks x seq_len_new x n_feats) matrix
        of chunked features (or masks)
    """
    # Pad
    if (np.shape(X)[0] % conf_dict["chunk_len"]):
        pad_len = conf_dict["chunk_len"] - (np.shape(X)[0] % conf_dict["chunk_len"])
        X = np.concatenate((X, -np.ones((pad_len, np.shape(X)[1]), dtype='float32')), axis=0)

    # Create chunks
    X = X.reshape((-1, conf_dict["chunk_len"], np.shape(X)[1]))

    return X


def create_chunks_phones(p, conf_dict):
    """
    Same as create_chunks, but for phones

    Args:
        p (np.array): (seq_len) array of phones across time steps
        conf_dict (dict): dictionary with configuration parameters

    Returns:
        p (np.array): (n_chunks x seq_len_new) matrix of chunked
        phones
    """
    # Pad
    if (np.shape(p)[0] % conf_dict["chunk_len"]):
        pad_len = conf_dict["chunk_len"] - (np.shape(p)[0] % conf_dict["chunk_len"])
        p = np.concatenate((p, np.zeros((pad_len,), dtype='long')), axis=0)

    # Create chunks
    p = p.reshape((-1, conf_dict["chunk_len"]))

    return p
