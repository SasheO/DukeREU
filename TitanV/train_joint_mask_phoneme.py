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


def train(model, optimizer, le_bpg, conf_dict, train_generator, torch_scaler):
    """ Train a mask estimation model over one epoch
    
    Args:
        model (torch.nn.Module): neural network model
        optimizer (optim.SGD): optimizer for mask estimation model
        conf_dict (dict): configuration parameters
        train_generator (DataLoader): data generator for training data
        torch_scaler (TorchStandardScaler): used to scale features
        
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
        X_batch = torch_scaler.transform((torch.from_numpy(X_batch)).to(device))
        y_batch = (torch.from_numpy(y_batch)).to(device)
        p_batch = (torch.from_numpy(p_batch)).to(device)

        # Get outputs
        bpg_outputs, bpg_specific_masks = model(X_batch)

        # Mixtures of experts loss
        if conf_dict["loss"] == "sig_approx":
            bpg_outputs = bpg_outputs.unsqueeze(2)
            y_pred = torch.sum(bpg_outputs * bpg_specific_masks, 3)
            loss_mask = (y_batch != -1) # valid T-F bins (i.e. not -1 padded)
            loss = loss_fn.sig_approx(X_batch, torch_scaler, y_pred, y_batch, loss_mask, 'mean')
        elif conf_dict["loss"] == "sig_approx_moe":
            loss_mask = (y_batch != -1)[:, :, 0] # valid T-F bins (i.e. not -1 padded)
            loss = loss_fn.sig_approx_moe(X_batch, torch_scaler, bpg_specific_masks, y_batch, bpg_outputs, loss_mask, 'mean')

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()


def validate(model, le_bpg, conf_dict, valid_generator, torch_scaler):
    """ Validate mask estimation model
    
    Args:
        model (torch.nn.Module): neural network model
        conf_dict (dict): configuration parameters
        valid_generator (DataLoader): data generator for validation data
        torch_scaler (TorchStandardScaler): used to normalize features
        
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
            X_val = torch_scaler.transform((torch.from_numpy(X_val)).to(device))
            y_val = (torch.from_numpy(y_val)).to(device)
            p_val = (torch.from_numpy(p_val)).to(device)

            # Get outputs and predictions
            bpg_outputs, bpg_specific_masks = model(X_val)

            # Mixture of experts loss
            if conf_dict["loss"] == "sig_approx":
                bpg_outputs = bpg_outputs.unsqueeze(2)
                y_pred = torch.sum(bpg_outputs * bpg_specific_masks, 3)
                loss_mask = (y_val != -1) # valid T-F bins (i.e. not -1 padded)
                loss = loss_fn.sig_approx(X_val, torch_scaler, y_pred, y_val, loss_mask, 'sum')
            elif conf_dict["loss"] == "sig_approx_moe":
                loss_mask = (y_val != -1)[:, :, 0] # valid T-F bins (i.e. not -1 padded)
                loss = loss_fn.sig_approx_moe(X_val, torch_scaler, bpg_specific_masks, y_val, bpg_outputs, loss_mask, 'sum')

            # Joint loss
            running_loss += loss.item()

            # Number of T-F bins to normalize, accounts for mask estimation and phoneme classification loss
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

    # Read in phoneme classification model
    phn_checkpoint = torch.load(conf_dict["bpg_model_dir"] + "/checkpoint.pt")
    phn_model = initialize_phn_network(conf_dict_phn)
    phn_model.load_state_dict(phn_checkpoint["model"])
    le = get_label_encoder(conf_dict["bpg"])    
        
    # Initialize the joint mask estimation model
    logging.info("Time: {}, Initializing model".format(datetime.now().strftime("%m/%d/%y %H:%M:%S")))
    model = initialize_network(conf_dict, conf_dict_phn)

    # Initialize each phoneme-specific model using trained mask estimation model
    pretrained_model_dir = os.path.join(conf_dict["pretrained_model_dir"])
    pretrained_model_filename = pretrained_model_dir + "/checkpoint.pt"

    pretrained_model = get_model_type(conf_dict)
    checkpoint = torch.load(pretrained_model_filename)
    pretrained_model.load_state_dict(checkpoint['model'], strict=False)
    
    for param_name, _ in model.named_parameters():
        if "mask" in param_name:
            temp = param_name.split(".")
            layer_mask_bpg = temp[0]
            layer = layer_mask_bpg.split("_")[0]
            param = temp[1]
    
            # Load weights
            setattr(getattr(model, layer_mask_bpg), param,
                    getattr(getattr(pretrained_model, layer), param))

    # Replace weights with trained phoneme classification network
    model.lstm_phn.weight_ih_l0.data.copy_(phn_model.lstm.weight_ih_l0.data)
    model.lstm_phn.weight_hh_l0.data.copy_(phn_model.lstm.weight_hh_l0.data)
    model.lstm_phn.bias_ih_l0.data.copy_(phn_model.lstm.bias_ih_l0.data)
    model.lstm_phn.bias_hh_l0.data.copy_(phn_model.lstm.bias_hh_l0.data)
    model.fc_phn.weight.data.copy_(phn_model.fc.weight.data)
    model.fc_phn.bias.data.copy_(phn_model.fc.bias.data)

    # For phoneme classification part of network, don't calculate the gradients (freezes weights)
    model.lstm_phn.weight_ih_l0.requires_grad = False
    model.lstm_phn.weight_hh_l0.requires_grad = False
    model.lstm_phn.bias_ih_l0.requires_grad = False
    model.lstm_phn.bias_hh_l0.requires_grad = False
    model.fc_phn.weight.requires_grad = False
    model.fc_phn.bias.requires_grad = False

    # Optimizer over mask estimation part of network
    if conf_dict["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])
    elif conf_dict["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=conf_dict["learning_rate"])    

    model.to(device)

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
            train(model, optimizer, le, conf_dict, training_generator, mask_scaler)

            # Validate
            valid_metrics = validate(model, le, conf_dict, valid_generator, mask_scaler)
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
    conf_file = "conf/ratio_mask_dp_fft/LSTM_1layer_but_rev_log_fft_8kutts_batch16_joint_moa_sigapprox.txt"

    # The name of the directory where the model will be stored
    model_name = "model3"

    # Train and validate model
    train_and_validate(conf_file, model_name)
