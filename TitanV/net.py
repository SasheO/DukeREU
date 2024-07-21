# net.py
# Author: Kevin Chu
# Last Modified: 7/21/2020 by Mezisashe Ojuba

import torch
import torch.nn as nn
import torch.nn.functional as F
from phone_mapping import get_label_encoder

class LSTMModelTransform(nn.Module): 
    '''
    think how you will change conf dict:
    
    "transform_by_bpg" = True (i.e. the mlp does a tranformation on each bpg)
    '''
    def __init__(self, conf_dict): ## TODO: how to add optimizers? maybe try test if it works then try to add different optimizers later
        super(LSTMModelTransform, self).__init__()
        # Label encoder for bpg
        self.le_bpg = get_label_encoder(conf_dict["bpg"])
    
        self.scale_model = MLP(input_size=len(self.le_bpg.classes_), output_size=conf_dict["num_coeffs"], activation=nn.ReLU)
        self.shift_model = MLP(input_size=len(self.le_bpg.classes_), output_size=conf_dict["num_coeffs"], activation=nn.LeakyReLU)
        
        self.lstm = LSTMModel(conf_dict)
        
        
    def forward(self, feature, phoneme):
        scales_pred = self.scale_model(phoneme)
        shifts_pred = self.shift_model(phoneme)
        
        if shifts_pred.shape != scales_pred.shape:
            raise Exception(f"MLP shift and scale models should have the same output shape: {shifts_pred.shape} vs {scales_pred.shape}")
        
        # construct affine matrix that scales and shifts      
        '''if len(shifts_pred.shape) == 1: # handle edge case where num_samples = 1
            num_samples = 1
        else:
            num_samples = shifts_pred.shape[0]
        m = torch.zeros(num_samples, scales_pred.shape[-1]+1, shifts_pred.shape[-1]+1)
        m[:,-1,-1] = 1
        for indx in range(m.shape[0]):
            m[indx,range(scales_pred.shape[-1]),range(scales_pred.shape[-1])] = scales_pred[indx]
            m[indx,:-1,-1] = shifts_pred[indx]'''
        
        # transform x and feed into output. did not construct an affine matrix anymore because that made computation more complex
        transformed_feature = feature*scales_pred+shifts_pred 
        
        output = self.lstm(transformed_feature)
        
        return output

class MLP(nn.Module):
    def __init__(self, input_size, output_size, activation): 
        super(MLP, self).__init__()
        self.sequential = nn.Sequential()
        self.sequential.add_module("dense1", nn.Linear(input_size, output_size))
        self.sequential.add_module("act1", activation())
        
    def forward(self, x):
        h = self.sequential(x)
        return h
    
class LSTMModel(nn.Module):

    def __init__(self, conf_dict):
        super(LSTMModel, self).__init__()

        # Stacked LSTMs
        if "num_layers" in conf_dict.keys():
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                num_layers=conf_dict["num_layers"], batch_first=True,
                                bidirectional=conf_dict["bidirectional"])
        # Default is one LSTM layer
        else:
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                batch_first=True, bidirectional=conf_dict["bidirectional"])

        # Batch norm
        #self.bn = nn.BatchNorm1d(conf_dict["num_hidden"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_outputs"])
        else:
            self.fc = nn.Linear(2*conf_dict["num_hidden"], conf_dict["num_outputs"])
            
        # Mask type
        self.mask = conf_dict["mask"]

        # Clip mask
        if "clip_mask" in conf_dict.keys():
            self.clip_mask = conf_dict["clip_mask"]

    def forward(self, x):        
        # Pass through LSTM layer
        h, (_, _) = self.lstm(x)

        # Batch norm
        #h = h.permute(0, 2, 1)
        #h = self.bn(h)
        #h = h.permute(0, 2, 1)

        # Pass hidden features to classification layer
        if "ratio_mask" in self.mask:
            out = torch.sigmoid(self.fc(h))
        elif "fft_mask" in self.mask:
            out = self.clip_mask*torch.sigmoid(self.fc(h))

        return out


class LSTMModelJointTrain(nn.Module):

    def __init__(self, conf_dict, conf_dict_phn):
        super(LSTMModelJointTrain, self).__init__()

        self.num_coeffs = conf_dict["num_coeffs"]
        self.num_outputs = conf_dict["num_outputs"]
        self.bpg = conf_dict["bpg"]
        self.le_bpg = get_label_encoder(conf_dict["bpg"])
        
        # PHONEME CLASSIFICATION
        # Stacked LSTMs
        if "num_layers" in conf_dict_phn.keys():
            self.lstm_phn = nn.LSTM(input_size=conf_dict_phn["num_coeffs"], hidden_size=conf_dict_phn["num_hidden"],
                                    num_layers=conf_dict_phn["num_layers"], batch_first=True,
                                    bidirectional=conf_dict_phn["bidirectional"])
        # Default is one LSTM layer
        else:
            self.lstm_phn = nn.LSTM(input_size=conf_dict_phn["num_coeffs"], hidden_size=conf_dict_phn["num_hidden"],
                                    batch_first=True, bidirectional=conf_dict_phn["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict_phn["bidirectional"]:
            self.fc_phn = nn.Linear(conf_dict_phn["num_hidden"], conf_dict_phn["num_classes"])
        else:
            self.fc_phn = nn.Linear(2*conf_dict_phn["num_hidden"], conf_dict_phn["num_classes"])

        # MASK ESTIMATION
        # Create each bpg expert
        for bpg in self.le_bpg.classes_:
            if "num_layers" in conf_dict.keys():
                setattr(self, "lstm_mask_" + bpg, nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                                          num_layers=conf_dict["num_layers"], batch_first=True,
                                                          bidirectional=conf_dict["bidirectional"]))
            else:
                setattr(self, "lstm_mask_" + bpg, nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                                          batch_first=True, bidirectional=conf_dict["bidirectional"]))

            # If bidirectional, double number of hidden units
            if not conf_dict["bidirectional"]:
                setattr(self, "fc_mask_" + bpg, nn.Linear(conf_dict["num_hidden"], conf_dict["num_outputs"]))
            else:
                setattr(self, "fc_mask_" + bpg, nn.Linear(2 * conf_dict["num_hidden"], conf_dict["num_outputs"]))

        # Mask type
        self.mask = conf_dict["mask"]

        # Clip mask
        if "clip_mask" in conf_dict.keys():
            self.clip_mask = conf_dict["clip_mask"]
            
    def forward(self, x):
        # PHONEME CLASSIFICATION
        # Inverse transform, apply mask, and normalize using scaler for phoneme classification network
        x_phn = x[:, :, -self.num_coeffs:] # only take current frame

        # Pass through LSTM layer
        h_phn, (_, _) = self.lstm_phn(x_phn)

        # Pass hidden features to classification layer
        bpg_outputs = torch.exp(F.log_softmax(self.fc_phn(h_phn), dim=2))

        # MASK ESTIMATION
        if x.is_cuda:
            device = torch.device("cuda:0")
        bpg_specific_masks = (torch.zeros(bpg_outputs.size()[0], bpg_outputs.size()[1], self.num_coeffs, bpg_outputs.size()[2])).to(device)
        
        for idx, bpg in enumerate(list(self.le_bpg.classes_)):
            # Get estimated mask for each expert
            lstm = getattr(self, "lstm_mask_" + bpg)
            h, (_, _) = lstm(x)

            fc = getattr(self, "fc_mask_" + bpg)
            if "ratio_mask" in self.mask:
                bpg_specific_masks[:, :, :, idx] = torch.sigmoid(fc(h))
            elif "fft_mask" in self.mask:
                bpg_specific_masks[:, :, :, idx] = self.clip_mask * torch.sigmoid(fc(h))

        return bpg_outputs, bpg_specific_masks


class LSTMModelPhnLoss(nn.Module):

    def __init__(self, conf_dict, conf_dict_phn):
        super(LSTMModelPhnLoss, self).__init__()

        # MASK ESTIMATION
        # Stacked LSTMs
        if "num_layers" in conf_dict.keys():
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                num_layers=conf_dict["num_layers"], batch_first=True,
                                bidirectional=conf_dict["bidirectional"])
        # Default is one LSTM layer
        else:
            self.lstm = nn.LSTM(input_size=conf_dict["num_features"], hidden_size=conf_dict["num_hidden"],
                                batch_first=True, bidirectional=conf_dict["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict["bidirectional"]:
            self.fc = nn.Linear(conf_dict["num_hidden"], conf_dict["num_outputs"])
        else:
            self.fc = nn.Linear(2 * conf_dict["num_hidden"], conf_dict["num_outputs"])

        # Mask type
        self.mask = conf_dict["mask"]

        # Clip mask
        if "clip_mask" in conf_dict.keys():
            self.clip_mask = conf_dict["clip_mask"]

        # PHONEME CLASSIFICATION
        # Stacked LSTMs
        if "num_layers" in conf_dict_phn.keys():
            self.lstm_phn = nn.LSTM(input_size=conf_dict_phn["num_coeffs"], hidden_size=conf_dict_phn["num_hidden"],
                                    num_layers=conf_dict_phn["num_layers"], batch_first=True,
                                    bidirectional=conf_dict_phn["bidirectional"])
        # Default is one LSTM layer
        else:
            self.lstm_phn = nn.LSTM(input_size=conf_dict_phn["num_coeffs"], hidden_size=conf_dict_phn["num_hidden"],
                                    batch_first=True, bidirectional=conf_dict_phn["bidirectional"])

        # If bidirectional, double number of hidden units
        if not conf_dict_phn["bidirectional"]:
            self.fc_phn = nn.Linear(conf_dict_phn["num_hidden"], conf_dict_phn["num_classes"])
        else:
            self.fc_phn = nn.Linear(2*conf_dict_phn["num_hidden"], conf_dict_phn["num_classes"])

    def forward(self, x, mask_scaler, phn_scaler):
        # MASK ESTIMATION
        # Pass through LSTM layer
        h, (_, _) = self.lstm(x)

        # Pass hidden features to classification layer
        if "ratio_mask" in self.mask:
            estimated_mask = torch.sigmoid(self.fc(h))
        elif "fft_mask" in self.mask:
            estimated_mask = self.clip_mask * torch.sigmoid(self.fc(h))

        # PHONEME CLASSIFICATION
        # Inverse transform, apply mask, and normalize using scaler for phoneme classification network
        x_phn = mask_scaler.inverse_transform(x)
        x_phn = x_phn[:, :, 0:estimated_mask.size()[2]] # only take current frame
        x_phn += 2*torch.log(estimated_mask)
        x_phn = phn_scaler.transform(x_phn)

        # Pass through LSTM layer
        h_phn, (_, _) = self.lstm_phn(x_phn)

        # Pass hidden features to classification layer
        out = F.log_softmax(self.fc_phn(h_phn), dim=2)

        return out


def initialize_weights(m):
    """ Initialize weights from Uniform(-0.1,0.1) distribution
    as was done in Graves and Schmidhuber, 2005
    
    Args:
        m
        
    Returns:
        none
    """
    a = -0.1
    b = 0.1

    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight.data, a=a, b=b)
        nn.init.uniform_(m.bias.data, a=a, b=b)

    if isinstance(m, nn.Conv2d):
        nn.init.uniform_(m.weight.data, a=a, b=b)
        nn.init.uniform_(m.bias.data, a=a, b=b)

    if isinstance(m, nn.RNN) or isinstance(m, nn.LSTM):
        nn.init.uniform_(m.weight_ih_l0, a=a, b=b)
        nn.init.uniform_(m.weight_hh_l0, a=a, b=b)
        nn.init.uniform_(m.bias_ih_l0, a=a, b=b)
        nn.init.uniform_(m.bias_hh_l0, a=a, b=b)


def initialize_weights_pretrained(m, mask_model, phn_model):
    """ Initialize weights from a pretrained model

    Args:
        m
        mask_model: pretrained mask estimation model
        phn_model: pretrained phoneme classification model

    Returns:
        none
    """
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            blah

        if isinstance(m, nn.LSTM):
            blah


def initialize_network(conf_dict, conf_dict_phn={}):
    """ Create network from configuration file and initialize
    weights

    Args:
        conf_dict (dict): conf file represented as a dict

    Returns:
        model (model): initialized model

    """
    # Instantiate the network
    model = get_model_type(conf_dict, conf_dict_phn)

    # Initialize weights
    model.apply(initialize_weights)

    return model


def get_model_type(conf_dict, conf_dict_phn={}):

    # Instantiate the network
    if conf_dict["model_type"] == "MLP":
        model = MLP(conf_dict)
    elif conf_dict["model_type"] == "CNN":
        model = CNN(conf_dict)
    elif conf_dict["model_type"] == "LSTM":
        if "loss" in conf_dict and (conf_dict["loss"] == "ce_phoneme" or conf_dict["loss"] == "ce_moa"):
            model = LSTMModelPhnLoss(conf_dict, conf_dict_phn)
        elif bool(conf_dict_phn):
            model = LSTMModelJointTrain(conf_dict, conf_dict_phn)
        elif "transform_by_bpg" in conf_dict and conf_dict["transform_by_bpg"] == True:
            model = LSTMModelTransform(conf_dict)
        else:
            model = LSTMModel(conf_dict)
        


    return model
