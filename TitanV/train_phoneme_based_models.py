import numpy as np
import random
import os
import os.path
import pickle
from pathlib import Path
from datetime import datetime

# Features
from conf import read_conf
from feature_extraction import fit_normalizer
from feature_extraction import fit_bpg_specific_normalizer
from feature_extraction import read_feat_file
from feature_extraction import TorchStandardScaler
from sklearn import preprocessing

# Labels
from phone_mapping import classes_to_sequences
from phone_mapping import get_label_list
from phone_mapping import get_label_encoder
from phone_mapping import map_phones

# Training and testing data
from file_loader import collate_fn
from file_loader import Dataset
from file_loader import read_feat_list
#import evaluate
import loss_fn

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Models
from net import initialize_network
from net import get_model_type

from tqdm import tqdm
import logging
import re
from shutil import copyfile

def get_device():
    """

    Returns:

    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


def train(experts, optimizers, le_bpg, conf_dict, train_generator, torch_scaler):
    """ Train a mask estimation model
    
    Args:
        experts (dict): contains phoneme-based models
        optimizers (dict): contains phoneme-based pytorch optimizers
        le_bpg (sklearn.preprocessing.LabelEncoder): label encoder for bpg
        conf_dict (dict): configuration parameters
        train_generator (DataLoader): data generator for training data
        torch.scaler (TorchStandardScaler): used to normalize features
        
    Returns:
        none
        
    """
    # Get device
    device = get_device()

    for X_batch, y_batch, p_batch in train_generator:
        # Move to GPU
        X_batch = torch_scaler.transform((torch.from_numpy(X_batch)).to(device))
        y_batch = (torch.from_numpy(y_batch)).to(device)
        p_batch = (torch.from_numpy(p_batch)).to(device)
        
        for bpg in list(experts.keys()):
            experts[bpg].train()
            optimizers[bpg].zero_grad()

            # Configure loss_mask to consider only the relevant bpg
            bpg_mask = (p_batch == ((le_bpg.transform([bpg])).astype('long')[0]))
            bpg_mask = bpg_mask.reshape(bpg_mask.size()[0], bpg_mask.size()[1], 1)
            loss_mask = bpg_mask * (y_batch != -1)

            # Only do forward/backward pass if bpg occurs in utterance
            if torch.sum(loss_mask) > 0:
                # Get outputs
                train_outputs = experts[bpg](X_batch)

                # Calculate loss
                if conf_dict["loss"] == "mse":
                    loss = loss_fn.mse(train_outputs, y_batch, loss_mask, 'mean')
                elif conf_dict["loss"] == "sig_approx":
                    loss = loss_fn.sig_approx(X_batch, torch_scaler, train_outputs, y_batch, loss_mask, 'mean')
                elif conf_dict["loss"] == "compress_sig_approx":
                    loss = loss_fn.compress_sig_approx(X_batch, torch_scaler, train_outputs, y_batch, conf_dict["power"], loss_mask, 'mean')

                # Backpropagate and update weights
                loss.backward()
                optimizers[bpg].step()


def validate(experts, le_bpg, conf_dict, valid_generator, torch_scaler):
    """ Validate mask estimation model
    
    Args:
        experts (dict): contains phoneme-based models
        le_bpg (sklearn.preprocessing.LabelEncoder): label encoder for bpg
        conf_dict (dict): configuration parameters
        file_list (list): list of feature files in training set
        model_dir (str): directory in which to save trained model
        
    Returns:
        metrics (dict): loss and accuracy averaged across batches
    
    """

    loss_dict = {}

    # Running values
    num_tf_bins = {}
    running_loss = {}

    # Get device
    device = get_device()

    # Evaluation mode
    for bpg in experts.keys():
        experts[bpg].eval()

        # Running values
        num_tf_bins[bpg] = 0
        running_loss[bpg] = 0

    with torch.no_grad():
        for X_val, y_val, p_val in valid_generator:
            # Move to GPU
            X_val = torch_scaler.transform((torch.from_numpy(X_val)).to(device))
            y_val = (torch.from_numpy(y_val)).to(device)
            p_val = (torch.from_numpy(p_val)).to(device)

            for bpg in list(experts.keys()):
                # Validate bpg experts separately
                bpg_mask = (p_val == ((le_bpg.transform([bpg])).astype('long')[0]))
                bpg_mask = bpg_mask.reshape(bpg_mask.size()[0], bpg_mask.size()[1], 1)
                loss_mask = bpg_mask * (y_val != -1)
                
                # Only validate if bpg occurs in utterance
                if torch.sum(loss_mask) > 0:
                    # Get outputs
                    outputs = experts[bpg](X_val)

                    # Calculate loss
                    if conf_dict["loss"] == "mse":
                        running_loss[bpg] += (loss_fn.mse(outputs, y_val, loss_mask, 'sum')).item()
                    elif conf_dict["loss"] == "sig_approx":
                        running_loss[bpg] += (loss_fn.sig_approx(X_val, torch_scaler, outputs, y_val, loss_mask, 'sum')).item()
                    elif conf_dict["loss"] == "compress_sig_approx":
                        running_loss[bpg] += (loss_fn.compress_sig_approx(X_val, torch_scaler, outputs, y_val, conf_dict["power"], loss_mask, 'sum')).item()
                        
                    num_tf_bins[bpg] += (torch.sum(loss_mask)).item()

    # Average loss over all batches
    for bpg in experts.keys():
        if num_tf_bins[bpg] > 0:
            loss_dict[bpg] = running_loss[bpg] / num_tf_bins[bpg]
        else:
            loss_dict[bpg] = -1

    return loss_dict


def train_and_validate(conf_file, model_name):
    """ Train and evaluate a mask estimation model

    Args:
        conf_file (str): txt file containing model info
        model_name (str): model name, used to distinguish between multiple instances
        of the same model type

    Returns
        none

    """
    # Read in conf file
    conf_dict = read_conf(conf_file)

    # Read in feature files
    train_list = read_feat_list(conf_dict["training"])
    valid_list = read_feat_list(conf_dict["development"])

    # Model directory - create new folder for each new instance of a model
    model_dir = os.path.join("exp")
    subdirs = conf_file.split("/")
    for i in range(1, len(subdirs)):
        model_dir = os.path.join(model_dir, subdirs[i].replace(".txt", ""))
    model_dir = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_dir):
        print("Directory already exists. Exiting program.")
        exit()
    else:
        Path(model_dir).mkdir(parents=True)

    # Initializing dictionaries for each expert
    experts = {}
    optimizers = {}
    min_loss_dict = {}
    loss_dict = {}
    device = get_device()

    phone_list = get_label_list("phone")

    # Convert phones to bpg
    if conf_dict["bpg"] == "phone":
        phone_list_as_bpg = phone_list
    else:
        phone_list_as_bpg = map_phones(phone_list, conf_dict)

    # Label encoder for bpg
    le_bpg = get_label_encoder(conf_dict["bpg"])

    # Configure log file
    logging.basicConfig(filename=model_dir+"/log", filemode="w", level=logging.INFO)

    # Get standard scalers from pretrained phoneme-independent model
    current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
    if "pretrained_model_dir" in conf_dict.keys():
        logging.info("{}, Reading Standard Scaler from phoneme-independent directory".format(current_time))
        with open(conf_dict["pretrained_model_dir"] + "/scaler.pickle", 'rb') as f:
            scaler = pickle.load(f)
    else:
        logging.info("{}, Fitting Standard Scaler".format(current_time))
        scaler = fit_normalizer(train_list, conf_dict)
    torch_scaler = TorchStandardScaler(scaler.mean_, scaler.var_, device)

    ########## CREATE MODELS ##########
    for bpg in le_bpg.classes_:
        # Save directory
        bpg_specific_model_dir = os.path.join(model_dir, bpg)
        Path(bpg_specific_model_dir).mkdir(parents=True)

        # Copy config file
        copyfile(conf_file, bpg_specific_model_dir + "/conf.txt")
            
        # Initialize experts using trained phoneme-independent model
        pretrained_model_dir = os.path.join(conf_dict["pretrained_model_dir"])
        pretrained_model_filename = pretrained_model_dir + "/checkpoint.pt"

        # Network
        experts[bpg] = get_model_type(conf_dict)
        checkpoint = torch.load(pretrained_model_filename)
        experts[bpg].load_state_dict(checkpoint['model'], strict=False)
        experts[bpg].to(device)

        # Optimizer
        if conf_dict["optimizer"] == "sgd":
            optimizers[bpg] = optim.SGD(experts[bpg].parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])
        elif conf_dict["optimizer"] == "adam":
            optimizers[bpg] = optim.Adam(experts[bpg].parameters(), lr=conf_dict["learning_rate"])
        optimizers[bpg].load_state_dict(checkpoint['optimizer'])

        # Save Standard Scaler to bpg-specific model directory
        scale_file = bpg_specific_model_dir + "/scaler.pickle"
        with open(scale_file, 'wb') as f:
            pickle.dump(scaler, f)

        # Training curves
        training_curves = bpg_specific_model_dir + "/training_curves"
        with open(training_curves, "w") as file_obj:
            file_obj.write("Epoch,Validation Loss\n")

        min_loss_dict[bpg] = float("inf")
        loss_dict[bpg] = []

    ########## TRAINING ##########
    # Training
    logging.info("Training")

    # Generators for loading utterances
    training_set = Dataset(train_list, conf_dict, le_bpg)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=conf_dict["batch_size"],
                                                     num_workers=4, collate_fn=collate_fn, shuffle=True)
    valid_set = Dataset(valid_list, conf_dict, le_bpg)
    valid_generator = torch.utils.data.DataLoader(valid_set, batch_size=conf_dict["batch_size"],
                                                  num_workers=4, collate_fn=collate_fn, shuffle=True)
        
    iterator = tqdm(range(conf_dict["num_epochs"]))
    for epoch in iterator:
        current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
        logging.info("Time: {}, Epoch: {}".format(current_time, epoch+1))

        # Train
        train(experts, optimizers, le_bpg, conf_dict, training_generator, torch_scaler)

        # Validate
        valid_loss = validate(experts, le_bpg, conf_dict, valid_generator, torch_scaler)

        for bpg in list(experts.keys()):
            loss_dict[bpg].append(valid_loss[bpg])

            bpg_specific_model_dir = os.path.join(model_dir, bpg)
            training_curves = bpg_specific_model_dir + "/training_curves"
            with open(training_curves, 'a') as file_obj:
                file_obj.write("{},{}\n".format(epoch+1, round(valid_loss[bpg], 3)))

            # Track the best model
            if valid_loss[bpg] < min_loss_dict[bpg]:
                min_loss_dict[bpg] = valid_loss[bpg]
                torch.save({'model': experts[bpg].state_dict(), 'optimizer': optimizers[bpg].state_dict()},
                           bpg_specific_model_dir +  "/checkpoint.pt")

            # Remove bpg-specific model if doesn't occur in validation set
            if loss_dict[bpg][-1] == -1:
                experts.pop(bpg)

            # Stop early and remove bpg-specific model if loss does not decrease over last 10 epochs
            if epoch >= 10:
                if loss_dict[bpg][-1] - loss_dict[bpg][-11] >= 0:
                    experts.pop(bpg)

        # Break if all bpg-specific models reach convergence
        if len(list(experts.keys())) == 0:
            logging.info("All models have converged. Stopping training.")
            iterator.close()
            break


if __name__ == '__main__':
    # USER INPUTS
    # The model architecture to use for training
    conf_file = "conf/ratio_mask_dp_fft/taslp2022_nonideal_phonemes/LSTM_1layer_but_rev_log_fft_batch16_moa_sigapprox.txt"

    # The name of the directory where the model will be stored
    model_name = "model1"

    # Train and validate model
    train_and_validate(conf_file, model_name)
