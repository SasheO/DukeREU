import numpy as np
import random
import os
import os.path
import pickle
from pathlib import Path
from datetime import datetime

# Features
from conf import read_conf
from feature_extraction import TorchStandardScaler
from feature_extraction import fit_normalizer
from feature_extraction import read_feat_file
from sklearn import preprocessing

# Labels
from phone_mapping import get_label_encoder

# Training and testing data
from file_loader_model_transform import collate_fn
from file_loader_model_transform import Dataset
from file_loader_model_transform import read_feat_list
import loss_fn

# PyTorch
import torch
import torch.nn.functional as F
import torch.optim as optim

# Models
from net import get_model_type
from net import initialize_network

from tqdm import tqdm
import logging
import re
from shutil import copyfile

import pickle

def get_device():
    """

    Returns:

    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


def train(model, optimizer, conf_dict, train_generator, torch_scaler):
    """ Train a mask estimation model over one epoch
    
    Args:
        model (torch.nn.Module): neural network model
        optimizer (optim.SGD): pytorch optimizer
        conf_dict (dict): configuration parameters
        train_generator (DataLoader): data generator for training data
        torch.scaler (TorchStandardScaler): used to normalize features
        
    Returns:
        none
        
    """
    # Training mode
    model.train()
    
    # Get device
    device = get_device()

    for X_batch, y_batch, _ in train_generator:
        optimizer.zero_grad()

        # Move to GPU
        X_batch = torch_scaler.transform((torch.from_numpy(X_batch)).to(device))
        y_batch = (torch.from_numpy(y_batch)).to(device)

        # Get outputs
        train_outputs = model(X_batch)

        # Loss mask
        loss_mask = (y_batch != -1)

        # Calculate loss
        if conf_dict["loss"] == "mse":
            loss = loss_fn.mse(train_outputs, y_batch, loss_mask, 'mean')
        elif conf_dict["loss"] == "sig_approx":
            loss = loss_fn.sig_approx(X_batch, torch_scaler, train_outputs, y_batch, loss_mask, 'mean')
        elif conf_dict["loss"] == "log_sig_approx":
            loss = loss_fn.log_sig_approx(X_batch, torch_scaler, train_outputs, y_batch, loss_mask, 'mean')
        elif conf_dict["loss"] == "compress_sig_approx":
            loss = loss_fn.compress_sig_approx(X_batch, torch_scaler, train_outputs, y_batch, conf_dict["power"], loss_mask, 'mean')
        elif conf_dict["loss"] == "gamma_sig_approx":
            loss = loss_fn.gamma_sig_approx(X_batch, torch_scaler, train_outputs, y_batch, conf_dict["power"], loss_mask, device, 'mean')
        elif conf_dict["loss"] == "mel_sig_approx":
            loss = loss_fn.mel_sig_approx(X_batch, torch_scaler, train_outputs, y_batch, loss_mask, device, 'mean')
        elif conf_dict["loss"] == "srmr":
            loss = loss_fn.srmr_loss(X_batch, torch_scaler, train_outputs, y_batch, loss_mask, device, 'mean')

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()


def validate(model, conf_dict, valid_generator, torch_scaler):
    """ Validate mask estimation model
    
    Args:
        model (torch.nn.Module): neural network model
        conf_dict (dict): configuration parameters
        valid_generator (DataLoader): data generator for validation data
        
    Returns:
        metrics (dict): loss and accuracy averaged across batches
    
    """
    # Evaluation mode
    model.eval()

    metrics = {}

    # Running values
    num_tf_bins = 0
    num_files = 0
    running_loss = 0

    # Get device
    device = get_device()

    with torch.no_grad():
        for X_val, y_val, _ in valid_generator:
            # Move to GPU
            X_val = torch_scaler.transform((torch.from_numpy(X_val)).to(device))
            y_val = (torch.from_numpy(y_val)).to(device)

            # Loss mask
            loss_mask = (y_val != -1)

            # Get outputs and predictions
            outputs = model(X_val)
        
            # Update running loss
            if conf_dict["loss"] == "mse":
                running_loss += (loss_fn.mse(outputs, y_val, loss_mask, "sum")).item()
            elif conf_dict["loss"] == "sig_approx":
                running_loss += (loss_fn.sig_approx(X_val, torch_scaler, outputs, y_val, loss_mask, 'sum')).item()
            elif conf_dict["loss"] == "log_sig_approx":
                running_loss += (loss_fn.log_sig_approx(X_val, torch_scaler, outputs, y_val, loss_mask, 'sum')).item()
            elif conf_dict["loss"] == "compress_sig_approx":
                running_loss += (loss_fn.compress_sig_approx(X_val, torch_scaler, outputs, y_val, conf_dict["power"], loss_mask, 'sum')).item()
            elif conf_dict["loss"] == "mel_sig_approx":
                running_loss += (loss_fn.mel_sig_approx(X_val, torch_scaler, outputs, y_val, loss_mask, device, 'sum')).item()
                loss_mask = loss_mask[:, :, 0:22]
            elif conf_dict["loss"] == "gamma_sig_approx":
                running_loss += (loss_fn.gamma_sig_approx(X_val, torch_scaler, outputs, y_val, conf_dict["power"], loss_mask, device, 'sum')).item()
                loss_mask = loss_mask[:, :, 0:22]
            elif conf_dict["loss"] == "srmr":
                running_loss += (loss_fn.srmr_loss(X_val, torch_scaler, outputs, y_val, loss_mask, device, 'sum')).item()
            
            num_tf_bins += (torch.sum(loss_mask)).item()
            num_files += (X_val.size()[0])

    # Average loss over all batches
    if conf_dict["loss"] == "srmr":
        metrics['loss'] = running_loss / num_files
    else:
        metrics['loss'] = running_loss / num_tf_bins

    return metrics


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
    model_dir = "/media/lab/Seagate Expansion Drive/Sashe/exp"
    subdirs = conf_file.split("/")
    for i in range(len(subdirs)):
        model_dir = os.path.join(model_dir, subdirs[i].replace(".txt", ""))
    model_dir = os.path.join(model_dir, model_name)
    
    if os.path.exists(model_dir):
        print(model_dir)
        print("Directory already exists. Exiting program.")
        exit()
    else:
        Path(model_dir).mkdir(parents=True)

    # Copy config file
    copyfile(conf_file, model_dir + "/conf.txt")
    #copyfile(conf_file, (conf_file.replace("conf/"+conf_dict["mask"]+"/", model_dir + "/")).replace(conf_file.split("/")[2], "conf.txt"))

    # Configure log file
    logging.basicConfig(filename=model_dir+"/log", filemode="w", level=logging.INFO)

    # Get standard scaler
    device = get_device()
    logging.info("Time: {}, Fitting Standard Scaler".format(datetime.now().strftime("%m/%d/%y %H:%M:%S")))
    scaler = fit_normalizer(train_list, conf_dict)
    with open(model_dir + "/scaler.pickle", 'wb') as f:
        pickle.dump(scaler, f)
    torch_scaler = TorchStandardScaler(scaler.mean_, scaler.var_, device)

    ########## CREATE MODEL ##########
    # Initialize the network
    logging.info("Time: {}, Initializing model".format(datetime.now().strftime("%m/%d/%y %H:%M:%S")))
    model = initialize_network(conf_dict)
    model.to(device)

    # Configure optimizer
    if conf_dict["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=conf_dict["learning_rate"], momentum=conf_dict["momentum"])
    elif conf_dict["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=conf_dict["learning_rate"])

    ########## TRAINING ##########
    # Training loss curves
    training_curves = model_dir + "/training_curves"
    with open(training_curves, "w") as file_obj:
        file_obj.write("Epoch,Validation Loss\n")

    logging.info("Training")
        
    # Generators for loading utterances
    training_set = Dataset(train_list, conf_dict) 
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=conf_dict["batch_size"],
                                                     num_workers=4, collate_fn=collate_fn, shuffle=True)
    valid_set = Dataset(valid_list, conf_dict)
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
            train(model, optimizer, conf_dict, training_generator, torch_scaler)

            # Validate
            valid_metrics = validate(model, conf_dict, valid_generator, torch_scaler)
            loss.append(valid_metrics["loss"])

            file_obj.write("{},{}\n".format(epoch+1, round(valid_metrics['loss'], 3)))

            # Track the best model and create checkpoint
            if valid_metrics['loss'] < min_loss:
                min_loss = valid_metrics["loss"]
                try:
                    '''torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_dir + "/checkpoint.pt")
                '''
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, '/media/lab/Seagate Expansion Drive/Sashe' + "/checkpoint.pt")
                except Exception as e:
                    print(1, e)
                    


            # Stop early if loss does not improve over last 10 epochs
            if epoch >= 10:
                if loss[-1] - loss[-11] >= 0:
                    logging.info("Detected minimum validation loss. Stopping early.")
                    iterator.close()
                    break


if __name__ == '__main__':
    # USER INPUTS
    # The model architecture to use for training
    conf_file = "conf/fft_mask_dp_fft/LSTMModelTransform.txt"

    # The name of the directory where the model will be stored
    model_name = "model1"

    # Train and validate model
    train_and_validate(conf_file, model_name)

