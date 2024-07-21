# MaskEstimationPytorch

## About
This repository contains code to develop time-frequency mask estimation models for speech enhancement in reverberant environments. The neural networks are implemented using PyTorch, and the features are currently calculated using MATLAB.

## Mask Estimation
A time-frequency (T-F) mask is a matrix of gain values that is multiplied by the T-F representation of a reverberant (and/or noisy) signal in order to obtain an estimate of the clean signal. The mask is typically based on the signal-to-noise ratio (SNR) or signal-to-reverberant ratio (SRR), where noise-dominated T-F units with a low SNR are multiplied by gain values near 0 (i.e. are attenuated) while speech-dominated T-F units with a high SNR are multiplied by gain values near 1 (i.e. are retained). Because the mask is unknown in a real-world scenario, an algorithm must be developed to estimate the mask using information extracted from the reverberant and/or noisy signal.

## Steps to Training and Testing Mask Estimation Models

### Training
1) In the feature_extraction directory, run `extractFeaturesTrainingData.m`. This is a MATLAB script that extracts features for the training and development sets. Refer to the README.md file and comments in the .m file for information on how to define the speech corpus and the RIR database to use for generating the training and development sets. The extracted features are stored in .txt files.<br/>
* IMPORTANT: Make sure to store the feature files in a server with lots of space (preferably not on a GPU). The total size of the feature files can easily be in the tens of GB.
2) Run `txt2npy.py`. This is a Python module that converts the feature files from .txt files to compressed .npz files, which allows the feature files to be read in much faster than .txt files, which subsequently allows the models to train much faster.<br/>
3) Create a model configuration file using either an existing model in the conf directory or by defining your own model. Additional detail is provided below in the section titled 'Configuration Files'.<br/>
4) Depending on the mask estimation model that you want to train, you will run one of the following Python scripts:<br/>
a) `train.py`: Trains phoneme-independent mask estimation model<br/>
b) `train_phoneme_based_models.py`: Trains voicing, MOA, and phoneme-specific mask estimation models<br/>
c) `train_joint_mask_phoneme.py`: Jointly trains voicing, MOA, and phoneme classification models with voicing, MOA, and phoneme-specific mask estimation models<br/>
d) `train_phnloss.py`: Trains mask estimation model using phoneme classification cross entropy as the loss<br/>

### Testing
The steps to test trained mask estimation models are very similar.<br/>
1) In the feature_extraction directory, run `extractFeaturesTestingData.m` to extract features for the testing set.<br/>
2) Run `txt2npy.py`.<br/>
3) Depending on the mask estimation model that you want to test, you will run one of the following Python scripts:<br/>
a) `run_evaluate.py`: Tests the phoneme-independent mask estimation model, and saves the estimated masks.<br/>
b) 'run_evaluate_phoneme_based_models.py`: tests the voicing, MOA, and phoneme-specific mask estimation models, and saves the estimated masks.<br/>
* IMPORTANT: Make sure to save the estimated masks in a server with lots of space (i.e. not on a GPU). The total size of the estimated mask files can easily be several GB.

### Objective Intelligibility
Note that when testing the mask estimation models, the provided scripts automatically calculate the loss on the test set. However, given that the majority of the loss functions are based on the mean squared error in either the mask domain or the signal domain, the loss does not necessarily predict speech intelligibility. Thus, various functions are provided to calculate objective intelligibility, and these functions are located in the objective_intelligibility directory.

* `calculateEcmMultipleConditions.m`: Calculates the envelope correlation-based measure for multiple test sets<br/>
* `calculateSrmrCiMultipleConditions.m`: Calculates the speech-to-reverberation modulation energy ratio tailored to CI users for multiple test sets. Note that this requires the SRMR_CI function the SRMR Toolbox (see https://github.com/MuSAELab/SRMRToolbox)<br/>
* `calculateStoiMultipleConditions.m`: Calculates the speech time objective intelligibility metric for multiple test sets. Note that this requires the stoi function from the STOI Toolbox<br/>
* `calculatePesqMultipleConditions.m`: Calculates the perceptual evaluation of speech quality of multiple test sets. Note that this requires the pesq function from the CD in Speech Enhancement: Theory and Practice<br/>

Note that the SRMR-CI, STOI, and PESQ calculations use different filterbanks than the default MAP in the Nucleus MATLAB Toolbox, so we must first resynthesize the signals in the time domain (using the phase of the reverberant signal) prior to calculating the objective metric.

## Models

### Configuration Files
The conf directory contains various configuration files that allow the user to define the model hyperparameters and the datasets used to develop the models. Here is an example of a conf file to train a phoneme-independent mask estimation model based on the LSTM architecture.

[Architecture]<br/>
model_type = LSTM<br/>
bidirectional = False<br/>
num_hidden = 128<br/>
num_layers = 1<br/>
num_outputs = 65<br/>
drop_feats = 0<br/>
drop_lstm = 0<br/>
drop_last_hidden = 0<br/>

[Training]<br/>
batch_size = 16<br/>
chunk_len = 1000<br/>
num_epochs = 250<br/>
loss = sig_approx<br/>
learning_rate = 1e-3<br/>
optimizer = adam<br/>

[Datasets]<br/>
training = data/ratio_mask_dp_fft/train_librispeech_sim_rev/log_fft.txt<br/>
development = data/ratio_mask_dp_fft/dev_librispeech_sim_rev/log_fft.txt<br/>

[Features]<br/>
feature_type = log_fft<br/>
num_coeffs = 65<br/>
window_size = 1<br/>
use_energy = False<br/>
deltas = False<br/>
deltaDeltas = False<br/>

[Labels]<br/>
mask = ratio_mask_dp_fft<br/>

The **Architecture** section contains detail regarding the model type, whether the model is unidirectional or bidirectional (for RNNs), and the hyperparameters of the model.
* `model_type`: indicates the model architecture (option: LSTM)<br/>
* `bidirectional`: uses bidirectional model if set to True, uses unidirectional model if set to False, valid only for RNNs (options: True or False)<br/>
* `num_hidden`: the number of hidden units (or memory blocks for LSTMs) (options: >= 1, integer)<br/>
* `num_layers`: the number of layers to include of the specified model type (options: >= 1, integer)<br/>
* `num_outputs`: the dimensionality of the predicted values (options: >= 1, integer)<br/>
* `drop_feats`:
* `drop_lstm`:
* `drop_last_hidden`:

The **Training** section contains detail regarding the batch size, maximum number of epochs, learning rate, momentum, classification task, and optimizer.
* `batch_size`: number of utterances to include in a batch (options: >= 1, integer)<br/>
* `chunk_len`: Used to split up long utterances into smaller chunks of certain length. Length should be provided in terms of the number of time frames (options: >= 1, integer)<br/>
* `num_epochs`: maximum number of epochs to run, but stops early if validation loss does not decrease over last 10 epochs (options: >= 1, integer)<br/>
* `loss`: Controls the loss function. Options:<br/>
    * `mse`: mean squared error between estimated and ideal mask<br/>
    * `sig_approx`: signal approximation error, which is the mean squared error between the signal enhanced using the estimated mask and the signal enhanced using the ideal mask<br/>
    * `log_sig_approx`: similar to signal approximation, but applies a log compression<br/>
    * `compress_sig_approx`: similar to signal approximation, but applies a power compression<br/>
    * `gamma_sig_approx`: similar to signal approximation, but calculates the loss after transforming the estimated enhanced signals and ideal enhanced signals to the gammatone representation<br/>
    * `mel_sig_approx`: similar to signal approximation, but calculates the loss after transforming the estimated enhanced signals and the ideal enhanced signals to the mel representation<br/>
* `learning_rate`: controls the learning rate (options: > 0)<br/>
* `optimizer`: controls the optimizer (options: adam, sgd)<br/>

The **Datasets** section contains the file paths to .txt files that contain lists of feature files.
* `training`: contains the list of feature files for the training set<br/>
* `development`: contains the list of feature files for the development set<br/>

The **Features** section contains metadata about the features used to train and test the models.
* `feature_type`: specifies the feature type (options: log_fft)<br/>
* `num_coeffs`: the dimensionality of the features (options: >= 1, integer)<br/>
* `window_size`: number of causal frames over which to splice (options: 1 = current frame, 2 = current frame + previous frames, etc.)<br/>
* `use_energy`: extract energy feature if set to True (options: True or False)<br/>
* `deltas`: extract deltas if set to True (options: True or False)<br/>
* `deltaDeltas`: extract delta-deltas if set to True (options: True or False)<br/>

The **Labels** section contains information about the label types.
* `mask`: specifies the T-F mask to estimate (options: ratio_mask_dp_fft, fft_mask_dp_fft)<br/>

For a phoneme group-based mask estimation model, the configuration files contain the following additional inputs:

[Hierarchical]<br/>
hierarchical = True<br/>
bpg = phoneme<br/>
bpg_model_dir = ../PhonemeClassificationPytorch/exp/phoneme/LSTM_sim_rev_fftspec_ci<br/>
phonemap = none<br/>

[Training]<br/>
pretrained_model_dir = exp/ratio_mask_dp_fft/LSTM_1layer_sim_rev_log_fft/model1<br/>

The **Architecture** section:
* `hierarchical`: whether to train a phoneme group-specific model (options: True = phoneme group-specific, False = phoneme-independent)<br/>
* `bpg`: the broad phonetic group (aka the phoneme group) used for training the mask estimation models (options: phone, phoneme, moa, bpg, vuv, moa_vuv)<br/>
* `bpg_model_dir`: the directory containing the trained phoneme group classifier<br/>
* `phonemap`: the file to use to map from phones (or phonemes) to phoneme group. Options:<br/>
    * For phoneme-specific models, set `phonemap` to none<br/>
    * For MOA-specific models, set `phonemap` to phones/phone_to_moa_timit.txt or phones/phone_to_moa_timit2.txt (note: recommend using phone_to_moa_timit2.txt to reproduce results from TASLP 2021 submission)<br/>
    * For voicing-specific models, set `phonemap` to phone_to_vuv.txt

### Relevant Modules
* `train.py`: trains phoneme-independent mask estimation model<br/>
* `train_phoneme_based_models.py`: trains phoneme-specific mask estimation models<br/>
* `evaluate.py`: evaluates phoneme-independent mask estimation model<br/>
* `evaluate_phoneme_based_models.py`: evaluates phoneme-specific mask estimation models<br/>
* `net.py`: contains model definitions<br/>
