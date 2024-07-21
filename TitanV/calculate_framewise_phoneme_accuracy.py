# External
import logging
import numpy as np
import os
import pickle
import sys
sys.path.insert(1, "../")
import torch
from tqdm import tqdm

# Internal
from file_loader_model_transform import read_feat_file
from file_loader_model_transform import read_feat_list
from phone_mapping import get_label_encoder
from phone_mapping import map_phones
from train import get_device
from conf import read_conf


def predict(model, le, conf_dict, file_list, est_mask_list, scale_file, mask_type):
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
    #logging.info("Testing model")

    # Track file name, true class, predicted class, and prob of predicted class
    summary = {"file": [], "y_true": [], "y_pred": [], "y_prob": []}

    # Get the device
    device = get_device()

    # Get scaler
    with open(scale_file, 'rb') as f:
        scaler = pickle.load(f)

    # Evaluation mode
    model.eval()
    print("Testing")

    with torch.no_grad():
        for i in tqdm(range(len(file_list))):
            #logging.info("Testing file {}".format(file_list[i]))

            # Extract features and labels for current file
            if mask_type == "none":
                X, _, phones = read_feat_file(file_list[i], conf_dict)
            else:
                if mask_type == "ideal":
                    X, mask, phones = read_feat_file(file_list[i], conf_dict)
                elif mask_type == "estimated":
                    X, _, phones = read_feat_file(file_list[i], conf_dict)
                    mask = np.loadtxt(est_mask_list[i], dtype='float32')

                # Convert log power spectrum to magnitude spectrum
                X = np.sqrt(np.exp(X))
                    
                # Masking
                X *= mask

                # Convert back to log power spectrum
                X = X**2
                X[X < np.finfo(np.float32).eps] = np.finfo(np.float32).eps
                X = np.log(X)

            # Log
            #X[X < np.finfo(np.float32).eps] = np.finfo(np.float32).eps
            #X = np.log(X)

            # Phone mapping
            if conf_dict["label_type"] == "phoneme":
                conf_dict["bpg"] = "phoneme"
                conf_dict["num_phonemes"] = 39
                conf_dict["phonemap"] = "phones/phone_map_Lee_Hon.txt"
                phones2 = phones
                #phones2 = map_phones(phones, conf_dict)
                #phones2 = np.array(phones2, dtype='object')
            elif conf_dict["label_type"] == "moa":
                phones2 = map_phones(phones, conf_dict)
                phones2 = np.array(phones2, dtype='object')

            # Normalize features
            X = scaler.transform(X)

            # Encode labels as integers
            phones2 = le.transform(phones2).astype('long')

            # Reshape to (num_batch, seq_len, num_feats/num_outs)
            X = np.reshape(X, (1, np.shape(X)[0], np.shape(X)[1]))

            # Move to GPU
            X = (torch.from_numpy(X)).to(device)
            phones2 = (torch.from_numpy(phones2)).to(device)

            # Get outputs and predictions
            outputs = model(X)
            y_prob = torch.exp(outputs)
            y_pred = torch.argmax(outputs, dim=2)
            y_pred = torch.squeeze(y_pred, 0)

            # Remove frames that correspond to h#
            #phones2 = phones2[np.argwhere(phones != 'h#')]
            #y_pred = y_pred[np.argwhere(phones != 'h#')]
            #y_prob = y_prob[np.argwhere(phones != 'h#'), :]

            # Update summary
            (summary['file']).append(file_list[i])
            (summary['y_true']).append(np.array(phones2.to('cpu')))
            (summary['y_pred']).append(np.array(y_pred.to('cpu')))
            (summary['y_prob']).append((y_prob.to('cpu')).detach().numpy())

    return summary


def test(conf_file, model_name, test_set, mask_type, bpg_label, phoneme_model_dir, save_dir):
    """ Make predictions and calculate performance metrics on
    the testing data.

    Args:
        conf_file (str): txt file containing model info
        model_idx (int): instance of the model
        test_set (str): specifies testing condition
        feat_type (str): mspec or mfcc

    Returns:
        none

    """
    # Import modules from PhonemeClassificationPytorch
    from PhonemeClassificationPytorch.net import get_model_type

    # Read configuration file
    conf_file_phn = phoneme_model_dir + "/conf.txt"
    conf_dict_phn = read_conf(conf_file_phn)

    # Load trained model
    model = get_model_type(conf_dict_phn)

    try:
        checkpoint = torch.load(phoneme_model_dir + "/checkpoint.pt")
        model.load_state_dict(checkpoint['model'], strict=False)
    except FileNotFoundError:
        model.load_state_dict(torch.load(phoneme_model_dir + "/model.pt"), strict=False)

    # Read configuration file
    conf_dict = read_conf(conf_file)

    # List of feature files for masking
    mask_feat_file = "data/" + conf_dict["mask"] + "/" + test_set + "/" + conf_dict["feature_type"] + ".txt"

    # Load trained model
    model_dir = os.path.join(save_dir, "exp", conf_dict["mask"], (conf_file.split("/")[2]).replace(".txt", ""),
                             model_name)

    # Scale file
    scale_file = phoneme_model_dir +  "/scaler.pickle"

    # Move to GPU
    device = get_device()
    model.to(device)

    # Directory where masks and phoneme accuracies are saved
    if "moa" in conf_file or "phoneme" in conf_file:
        mask_dir = os.path.join(model_dir, "estimated_masks", test_set)
        results_dir = os.path.join(model_dir, "results", test_set)
    else:
        mask_dir = os.path.join(model_dir, "estimated_masks", test_set)
        results_dir = os.path.join(model_dir, "results", test_set)

    # Read in list of feature files for masking
    mask_feat_list = read_feat_list(mask_feat_file)

    # List of estimated masks
    est_mask_list = list(map(lambda a: mask_dir + a.split(test_set)[1], mask_feat_list))

    # Get predictions
    summary = predict(model, get_label_encoder(conf_dict_phn["label_type"]), conf_dict_phn, mask_feat_list, est_mask_list, scale_file, mask_type)

    # Accuracy
    summary['y_true'] = np.concatenate(summary['y_true'])
    summary['y_pred'] = np.concatenate(summary['y_pred'])
    print(results_dir)
    calculate_performance_metrics(summary, results_dir, mask_type)


def calculate_performance_metrics(summary, results_dir, mask_type):
    # Phoneme accuracy
    phoneme_accuracy = np.sum(summary['y_true'] == summary['y_pred']) / len(summary['y_true'])
    print(phoneme_accuracy)

    if mask_type == "none":
        phoneme_accuracy_file = results_dir + "/phoneme_accuracy_unmitigated.txt"
    elif mask_type == "estimated":
        phoneme_accuracy_file = results_dir + "/phoneme_accuracy.txt"
    elif mask_type == "ideal":
        phoneme_accuracy_file = results_dir + "/phoneme_accuracy_ideal.txt"

    with open(phoneme_accuracy_file, 'w') as f:
        f.write("Accuracy: " + str(phoneme_accuracy) + "\n")
        f.write("Correct: " + str(np.sum(summary['y_true'] == summary['y_pred'])) + "\n")
        f.write("Total: " + str(len(summary['y_true'])) + "\n")


if __name__ == '__main__':
    # Inputs
    save_dir = "/media/batcave/personal/chu.kevin/TitanV/MaskEstimationPytorch"
    conf_file = "conf/ratio_mask_dp_fft/taslp2022_nonideal_phonemes/LSTM_1layer_but_rev_log_fft_8kutts_batch16_sigapprox.txt"
    model_name = "LSTM_1layer_but_rev_log_fft_batch16_sigapprox/model1"
    test_set = "test_hint_aula_carolina_1_1_4_90_3"
    mask_type = "none"
    bpg_label = "known"
    phoneme_model_dir = "/home/lab/Kevin/PhonemeClassificationPytorch/exp/phoneme/LSTM_but_rev_fftspec_ci/librispeech_rev_arpabet"

    test(conf_file, model_name, test_set, mask_type, bpg_label, phoneme_model_dir, save_dir)
