from train import get_device
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from feature_extraction import read_feat_file
from feature_extraction import TorchStandardScaler
import numpy as np
from conf import read_conf
import os
from file_loader_model_transform import read_feat_list,encode_phones
from pathlib import Path
import logging
from net import get_model_type
import loss_fn
# Labels
from phone_mapping import get_label_encoder

def predict(model, conf_dict, file_list, scale_file):
    """ Test phoneme classification model

    Args:
        model (torch.nn.Module): neural network model
        le (sklearn.preprocessing.LabelEncoder): encodes string labels as integers
        conf_dict (dict): configuration parameters
        file_list (list): files in the test set
        scaler (StandardScaler): scales features to zero mean unit variance

    Returns:
        summary (dict): dictionary containing file name, true class
        predicted class, and probability of predicted class

    """
    logging.info("Testing model")

    # Track file name, true class, predicted class, and prob of predicted class
    summary = {"file": [], "y_true": [], "y_pred": [], "loss": []}

    # Get the device
    device = get_device()

    # Get scaler
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    torch_scaler = TorchStandardScaler(scaler.mean_, scaler.var_, device)

    # Evaluation mode
    model.eval()
    print("Testing")

    with torch.no_grad():
        for i in tqdm(range(len(file_list))):
            logging.info("Testing file {}".format(file_list[i]))

            # Extract features and labels for current file
            X, y, phones = read_feat_file(file_list[i], conf_dict)

            # encode phones
            le_bpg = get_label_encoder(conf_dict["bpg"])
            _phones = encode_phones(phones, le_bpg)

            # Normalize features
            X = scaler.transform(X)

            # Reshape to (num_batch, seq_len, num_feats/num_out)
            X = np.reshape(X, (1, np.shape(X)[0], np.shape(X)[1]))
            y = np.reshape(y, (1, np.shape(y)[0], np.shape(y)[1]))
            _phones = np.reshape(_phones, (1, np.shape(_phones)[0], np.shape(_phones)[1]))
            

            # Move to GPU
            X = (torch.from_numpy(X)).to(device)
            y = (torch.from_numpy(y)).to(device)
            _phones = (torch.from_numpy(_phones)).to(device)

            # Get predictions
            output = model(X, _phones.float())

            # If tuple, use moe model
            if type(output) is tuple:
                bpg_outputs = (output[0]).unsqueeze(2)
                bpg_specific_masks = output[1]
                y_pred = torch.sum(bpg_outputs * bpg_specific_masks, 3)
            # If tensor, use phoneme-independent model
            else:
                y_pred = output

            # Ignore h#
            loss_mask = (phones != "sil")
            loss_mask = np.tile(np.reshape(loss_mask, (1, len(loss_mask), 1)), (1, 1, y.size()[2]))
            loss_mask = torch.from_numpy(loss_mask).to(device)

            # Get loss
            loss = loss_fn.sig_approx(X, torch_scaler, y_pred, y, loss_mask, 'mean')

            # Reshape
            y = torch.reshape(y, (y.size()[1], y.size()[2]))
            y_pred = torch.reshape(y_pred, (y_pred.size()[1], y_pred.size()[2]))

            # Update summary
            (summary['file']).append(file_list[i])
            (summary['y_true']).append(np.array(y.to('cpu')))
            (summary['y_pred']).append(np.array(y_pred.to('cpu')))
            (summary['loss']).append(np.array(loss.to('cpu')))

    summary['loss'] = np.array(summary['loss'])
    return summary


def test(conf_file, model_name, test_set, save_dir):
    """ Make predictions and calculate performance metrics on
    the testing data.

    Args:
        conf_file (str): txt file containing model info
        model_name (str): folder where model is saved
        test_set (str): specifies testing condition
        save_dir (str): base directory in which to save masks

    Returns:
        none

    """

    # Read configuration file
    conf_dict = read_conf(conf_file)

    # List of feature files for the testing data
    test_feat_list = "data/" + conf_dict["mask"] + "/" + test_set + "/" + conf_dict["feature_type"] + ".txt"

    # Get directory where trained model is located
    model_dir = os.path.join(save_dir, "exp")
    model_dir = os.path.join(model_dir, "conf")
    subdirs = conf_file.split("/")
    for i in range(1, len(subdirs)):
        model_dir = os.path.join(model_dir, subdirs[i].replace(".txt", ""))
    model_dir = os.path.join(model_dir, model_name)

    # Get blank model
    model = get_model_type(conf_dict)
    
    # Load in the model parameters
    # uncomment this below
    # checkpoint = torch.load(model_dir + "/checkpoint.pt")
    
    checkpoint = torch.load(model_dir+"/checkpoint.pt")
    model.load_state_dict(checkpoint['model'], strict=False)
    '''try:
        if "pretrained_model_dir" in conf_dict:
            checkpoint = torch.load(conf_dict["pretrained_model_dir"] + "/checkpoint.pt")
        else:
            checkpoint = torch.load(model_dir + "/checkpoint.pt")
        model.load_state_dict(checkpoint['model'], strict=False)
    except FileNotFoundError:
        model.load_state_dict(torch.load(model_dir + "/model.pt"), strict=False)'''

    # Move to GPU
    device = get_device()
    model.to(device)

    # Directory in which to save decoding results
    results_dir = os.path.join(model_dir, "results", test_set)

    try:
        Path(results_dir).mkdir(parents=True)
    except FileExistsError:
        print("File exists: " + results_dir + ". Overwriting existing files")

    # Directory in which to save masks
    mask_dir = os.path.join(model_dir, "estimated_masks", test_set)

    try:
        Path(mask_dir).mkdir(parents=True)
    except FileExistsError:
        print("File exists: " + mask_dir + ". Overwriting existing files")

    # Configure log file
    logging.basicConfig(filename=results_dir+"/log", filemode="w", level=logging.INFO)

    # Read in list of feature files
    test_list = read_feat_list(test_feat_list)

    # File containing StandardScaler computed based on the training data
    if "pretrained_model_dir" in conf_dict:
        scale_file = conf_dict["pretrained_model_dir"] + "/scaler.pickle"
    else:
        scale_file = model_dir + "/scaler.pickle"

    # Get predictions
    summary = predict(model, conf_dict, test_list, scale_file)

    save_metrics(summary, test_set, results_dir)
    save_masks(summary, test_set, mask_dir)


def save_metrics(summary, test_set, results_dir):
    """ Saves metrics for estimated masks

    Args:
        summary: dictionary with true and predicted masks and loss
        test_set (str): name of test set
        results_dir (str): directory in which to save metrics

    Returns:
        none
    """
    out_file = results_dir + "/loss.txt"
    np.savetxt(out_file, summary['loss'], fmt='%.4f', delimiter='\n')


def save_masks(summary, test_set, mask_dir):
    """ Saves estimated masks

    Args:
        summary: dictionary with true and predicted masks and loss
        test_set (str): name of test set
        mask_dir (str): directory in which to save masks

    Returns:
        none

    """

    for i, file in enumerate(summary['file']):
        # File where estimated masks are saved
        timit_file = file.split(test_set)[1]
        out_file = mask_dir + timit_file

        # Create directory
        out_dir = ("/").join(out_file.split("/")[0:-1])
        try:
            Path(out_dir).mkdir(parents=True)
        except FileExistsError:
            a = 1
            #print("File exists: " + out_dir + ". Overwriting existing files")

        np.savetxt(out_file, summary['y_pred'][i], fmt='%.4f', delimiter=' ')
