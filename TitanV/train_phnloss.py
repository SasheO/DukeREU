import numpy as np
import random
import os
import os.path
import pickle
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(1, "../")

# Features
from conf import read_conf
from feature_extraction import TorchStandardScaler
from feature_extraction import fit_normalizer
from feature_extraction import read_feat_file
from sklearn import preprocessing

# Labels
from phone_mapping import get_label_encoder
from phone_mapping import get_moa_list
from phone_mapping import map_phones

# Training and testing data
from file_loader import collate_fn
from file_loader import Dataset
from file_loader import read_feat_list
import loss_fn

# PyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim

# Models
from net import get_model_type
from net import initialize_network
from net import initialize_weights

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


def train(model, optimizer, conf_dict, train_generator, mask_scaler, phn_scaler):
    """ Train a mask estimation model over one epoch
    
    Args:
        model (torch.nn.Module): neural network model
        optimizer (optim.SGD): optimizer for mask estimatino model
        conf_dict (dict): configuration parameters
        train_generator (DataLoader): data generator for training data
        mask_scaler (TorchStandardScaler): used to normalize features for mask estimation
        phn_scaler (TorchStandardScaler): used to normalize features for phn classification
        
    Returns:
        none
        
    """
    # Training mode
    model.train()

    # Get device
    device = get_device()

    for X_batch, y_batch, p_batch in train_generator:
        optimizer.zero_grad()

        # Move to GPU
        X_batch = mask_scaler.transform((torch.from_numpy(X_batch)).to(device))
        y_batch = (torch.from_numpy(y_batch)).to(device)
        p_batch = (torch.from_numpy(p_batch)).to(device)

        # Get outputs
        train_outputs = model(X_batch, mask_scaler, phn_scaler)
        train_outputs = train_outputs.permute(0, 2, 1)

        # Loss mask - only calculate loss over valid region of the utterance (i.e. not -1 padded)
        loss_mask = torch.prod(y_batch != -1, axis=2)
        loss = torch.sum(loss_mask * F.nll_loss(train_outputs, p_batch, reduction='none'))/torch.sum(loss_mask)

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()


def validate(model, conf_dict, valid_generator, mask_scaler, phn_scaler):
    """ Validate mask estimation model
    
    Args:
        model (torch.nn.Module): neural network model
        conf_dict (dict): configuration parameters
        valid_generator (DataLoader): data generator for validation data
        mask_scaler (TorchStandardScaler): used to normalize features for mask estimation
        phn_scaler (TorchStandardScaler): used to normalize features for phn classification
        
    Returns:
        metrics (dict): loss and accuracy averaged across batches
    
    """
    # Evaluation mode
    model.eval()

    metrics = {}

    # Running values
    num_tf_bins = 0
    running_loss = 0

    # Get device
    device = get_device()

    with torch.no_grad():
        for X_val, y_val, p_val in valid_generator:
            # Move to CPU
            X_val = mask_scaler.transform((torch.from_numpy(X_val)).to(device))
            y_val = (torch.from_numpy(y_val)).to(device)
            p_val = (torch.from_numpy(p_val)).to(device)

            # Get outputs and predictions
            outputs = model(X_val, mask_scaler, phn_scaler)
            outputs = outputs.permute(0, 2, 1)

            # Loss mask
            loss_mask = torch.prod(y_val != -1, axis=2)
            running_loss += (torch.sum(loss_mask * F.nll_loss(outputs, p_val, reduction='none'))).item()
            num_tf_bins += (torch.sum(loss_mask)).item()

        # Average loss over all batches
        metrics['loss'] = running_loss / num_tf_bins

        return metrics

    
def train_and_validate(conf_file, model_name):
    """ Train and evaluate a mask estimation model

    Args:
        conf_file (str): txt file containing model info
        model_name (str): model name, directory in which to save model

    Returns
        none

    """
    # Import modules from PhonemeClassificationPytorch
    from PhonemeClassificationPytorch.net import initialize_network as initialize_phn_network

    # Read in mask estimation conf file
    conf_dict = read_conf(conf_file)

    # Read in feature files
    train_list = read_feat_list(conf_dict["training"])
    valid_list = read_feat_list(conf_dict["development"])

    # Model directory - create new folder for each new instance of a model
    model_dir = os.path.join("exp", conf_dict["mask"], (conf_file.split("/")[2]).replace(".txt", ""), model_name)
    if os.path.exists(model_dir):
        print("Directory already exists. Exiting program.")
        exit()
    else:
        Path(model_dir).mkdir(parents=True)

    # Copy config file
    copyfile(conf_file, (conf_file.replace("conf/"+conf_dict["mask"]+"/", model_dir + "/")).replace(conf_file.split("/")[2], "conf.txt"))

    # Configure log file
    logging.basicConfig(filename=model_dir+"/log", filemode="w", level=logging.INFO)
    
    # Get standard scaler
    device = get_device()
    logging.info("Time: {}, Fitting Standard Scaler".format(datetime.now().strftime("%m/%d/%y %H:%M:%S")))
    scaler = fit_normalizer(train_list, conf_dict)
    with open(model_dir + "/scaler.pickle", 'wb') as f:
        pickle.dump(scaler, f)
    mask_scaler = TorchStandardScaler(scaler.mean_, scaler.var_, device)

    ########## CREATE MODEL ##########
    # Scaler for phoneme classification model
    conf_dict_phn = read_conf(conf_dict["bpg_model_dir"] + "/conf.txt")
    with open(conf_dict["bpg_model_dir"] + "/scaler.pickle", 'rb') as f:
        temp_scaler = pickle.load(f)
    phn_scaler = TorchStandardScaler(temp_scaler.mean_, temp_scaler.var_, device)
    
    # Read in phoneme classification model
    phn_checkpoint = torch.load(conf_dict["bpg_model_dir"] + "/checkpoint.pt")
    phn_model = initialize_phn_network(conf_dict_phn)
    phn_model.load_state_dict(phn_checkpoint["model"])
    le = get_label_encoder(conf_dict["bpg"])
    
    # Initialize the mask estimation model
    logging.info("Time: {}, Initializing model".format(datetime.now().strftime("%m/%d/%y %H:%M:%S")))
    model = initialize_network(conf_dict, conf_dict_phn)

    # Replace weights with trained phoneme classification network
    model.lstm_phn.weight_ih_l0.data.copy_(phn_model.lstm.weight_ih_l0.data)
    model.lstm_phn.weight_hh_l0.data.copy_(phn_model.lstm.weight_hh_l0.data)
    model.lstm_phn.bias_ih_l0.data.copy_(phn_model.lstm.bias_ih_l0.data)
    model.lstm_phn.bias_hh_l0.data.copy_(phn_model.lstm.bias_hh_l0.data)
    model.fc_phn.weight.data.copy_(phn_model.fc.weight.data)
    model.fc_phn.bias.data.copy_(phn_model.fc.bias.data)

    model.to(device)

    # Only update parameters of the mask estimation network
    parameters_to_update = [{'params': model.lstm.parameters(), "lr": conf_dict["learning_rate"]},
                            {'params': model.fc.parameters(), "lr": conf_dict["learning_rate"]}]
                            #{'params': model.lstm_phn.parameters(), "lr": conf_dict_phn["learning_rate"]},
                            #{'params': model.fc_phn.parameters(), "lr": conf_dict_phn["learning_rate"]}]

    # Configure optimizer
    if conf_dict["optimizer"] == "sgd":
        optimizer = optim.SGD(parameters_to_update, lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])
    elif conf_dict["optimizer"] == "adam":
        optimizer = optim.Adam(parameters_to_update, lr=conf_dict["learning_rate"])

    ########## TRAINING ##########
    # Training loss curves
    training_curves = model_dir + "/training_curves"
    with open(training_curves, "w") as file_obj:
        file_obj.write("Epoch,Validation Loss\n")

    logging.info("Training")

    # Generators for loading utterances
    training_set = Dataset(train_list, conf_dict, le)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=conf_dict["batch_size"],
                                                     num_workers=4, collate_fn=collate_fn, shuffle=True)
    valid_set = Dataset(valid_list, conf_dict, le)
    valid_generator = torch.utils.data.DataLoader(valid_set, batch_size=conf_dict["batch_size"],
                                                  num_workers=4, collate_fn=collate_fn, shuffle=True)

    # Variables used to track minimum loss
    min_loss = float("inf")
    loss = []
    
    iterator = tqdm(range(conf_dict["num_epochs"]))
    
    for epoch in iterator:
        with open(training_curves, "a") as file_obj:
            current_time = datetime.now().strftime("%m/%d/%y %H:%M:%S")
            logging.info("Time: {}, Epoch: {}".format(current_time, epoch+1))

            # Train
            train(model, optimizer, conf_dict, training_generator, mask_scaler, phn_scaler)

            # Validate
            valid_metrics = validate(model, conf_dict, valid_generator, mask_scaler, phn_scaler)
            loss.append(valid_metrics["loss"])

            file_obj.write("{},{}\n".format(epoch+1, round(valid_metrics['loss'], 3)))

            # Track the best model
            if valid_metrics['loss'] < min_loss:
                min_loss = valid_metrics["loss"]
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           model_dir + "/checkpoint.pt")

            # Stop early if loss does not improve over last 10 epochs
            if epoch >= 10:
                if loss[-1] - loss[-11] >= 0:
                    logging.info("Detected minimum validation loss. Stopping early.")
                    iterator.close()
                    break


if __name__ == '__main__':
    # USER INPUTS
    # The model architecture to use for training
    conf_file = "conf/ratio_mask_dp_fft/LSTM_1layer_sim_rev_log_fft_batch16_phnloss.txt"

    # The name of the directory where the model will be stored
    model_name = "model1"

    # Train and validate model
    train_and_validate(conf_file, model_name)
