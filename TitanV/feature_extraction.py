# featureExtraction.py
# Author: Kevin Chu
# Last Modified: 06/11/2020

# External
import numpy as np
from sklearn import preprocessing
import torch

# Internal
from file_loader_model_transform import collate_fn
from file_loader_model_transform import Dataset
from file_loader_model_transform import read_feat_file
from phone_mapping import get_label_list
from phone_mapping import map_phones


class TorchStandardScaler:
    """
    Standard scaler for PyTorch tensors
    """

    def __init__(self, mean, var, device):
        self.mean = (torch.tensor(mean.astype('float32')).unsqueeze(0)).to(device)
        self.var = (torch.tensor(var.astype('float32')).unsqueeze(0)).to(device)

    def transform(self, x):
        """ Scales features to zero mean and unit variance
        
        Args:
            x (torch.Tensor): (batch x seq_len x nfeats)

        Returns:
            x_norm (torch.Tensor): scaled features
        """
        x_norm = (x - self.mean)/torch.sqrt(self.var)

        return x_norm

    def inverse_transform(self, x_norm):
        """ Undo zero mean and unit variance

        Args:
            x_norm (torch.Tensor): normalized features

        Returns:
            x (torch.Tensor): un-normalized features
        """
        x = (x_norm * torch.sqrt(self.var)) + self.mean

        return x


def fit_normalizer(file_list, conf_dict):
    """ Fits feature normalizer using list of files

    Args:
        file_list (list): list of training files
        conf_dict (dict): configuration dictionary

    Returns:
        scaler (StandardScaler): scaler estimated on training data
    """

    # Instantiate scaler
    scaler = preprocessing.StandardScaler()

    # Genterator for loading utterances
    data_set = Dataset(file_list, conf_dict)
    data_generator = torch.utils.data.DataLoader(data_set, batch_size=conf_dict["batch_size"],
                                                 num_workers=4, collate_fn=collate_fn, shuffle=True)

    # Read in one batch at a time and update scaler
    for X_batch, _, _ in data_generator:
        # Reshape
        X_batch = np.reshape(X_batch, (np.shape(X_batch)[0]*np.shape(X_batch)[1], np.shape(X_batch)[2]))

        # Update scaler using valid indices (i.e. not -1)
        scaler.partial_fit(X_batch[np.where(np.prod(X_batch != -1, axis=1))])

    return scaler


def fit_bpg_specific_normalizer(file_list, conf_dict, bpg_type):
    """ Fits bpg-specific feature normalizer using list of files

    Args:
        file_list (list): list of training files
        conf_dict (dict): configuration dictionary
        bpg_type (str): broad phonetic group type

    Returns:
        scalers (dict): bpg specific scalers estimated on training data
    """

    bpg_list = get_label_list(bpg_type)

    # Instantiate scaler for each bpg
    scalers = {}
    for bpg in bpg_list:
        scalers[bpg] = preprocessing.StandardScaler()

    # Iteratively update standard scaler for each bpg
    for file in file_list:
        # Get features
        X, _, phones = read_feat_file(file, conf_dict)

        # Mapping
        phones = np.array(map_phones(list(phones), conf_dict))

        # Update standard scaler for each bpg
        for bpg in np.unique(phones):
            is_bpg = (phones == bpg)
            scalers[bpg].partial_fit(X[is_bpg, :])

    return scalers
