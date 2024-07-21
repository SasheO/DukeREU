# PyTorch
from gammatone.fftweight import fft_weights
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import objective_intelligibility.srmr


def mse(y_pred, y_true, loss_mask, avg_or_sum):
    """
    Calculate mean squared error of estimated mask with respect
    to ideal mask. This function calculates the loss over all
    valid T-F bins (i.e. those that are not padded with -1's).

    Args:
        y_pred (3D tensor): predicted mask
        y_true (3D tensor): ideal mask
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: loss between estimated and true mask
    """
    # Calculate sum or mean over time, frequency, and batch
    if avg_or_sum == "sum":
        loss = torch.sum(loss_mask * F.mse_loss(y_pred, y_true, reduction='none'))
    elif avg_or_sum == "mean":
        loss = torch.sum(loss_mask * F.mse_loss(y_pred, y_true, reduction='none'))/(torch.sum(loss_mask))
    
    return loss


def sig_approx(X, torch_scaler, y_pred, y_true, loss_mask, avg_or_sum):
    """
    Calculate signal approximation loss, which is defined as
    the mean squared error between the T-F signal enhanced using
    the estimated mask and the signal enhanced using the ideal
    mask.

    Args:
        X (3D tensor): normalized features
        torch_scaler (TorchStandardScaler)
        y_pred (3D tensor): predicted mask
        y_true (3D tensor): ideal mask
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: signal approximation loss
    """
    # Un-normalize features
    X = torch_scaler.inverse_transform(X)

    # Take features corresponding to current time window
    X = X[:, :, -y_true.size()[2]:]

    # Transform log power spectrum features back to magnitude spectrum
    # to calculate the loss
    X[loss_mask] = torch.sqrt(torch.exp(X[loss_mask]))

    # Compute enhanced magnitude spectra
    enh_est = y_pred*X
    enh_ideal = y_true*X

    # Numerical stability
    enh_est[enh_est < 0] = 0
    enh_ideal[enh_ideal < 0] = 0

    if avg_or_sum == "sum":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est, enh_ideal, reduction='none'))
    elif avg_or_sum == "mean":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est, enh_ideal, reduction='none'))/(torch.sum(loss_mask))

    return loss


def sig_approx_moe(X, torch_scaler, y_bpg_specific, y_true, bpg_outputs, loss_mask, avg_or_sum):
    """
    Mixture of experts loss with signal approximation

    Args:
        X (3D tensor): normalized features
        torch_scaler (TorchStandardScaler)
        y_bpg_specific (4D tensor): bpg-specific masks
        y_true (3D tensor): ideal mask
        bpg_outputs (3D tensor): softmax probabilities from phn classif model
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: mixture of experts loss
    """
    # Un-normalize features
    X = torch_scaler.inverse_transform(X)

    # Take features corresponding to current time window
    X = X[:, :, -y_true.size()[2]:]

    # Transform log power spectrum features back to magnitude spectrum
    # to calculate the loss
    X = torch.sqrt(torch.exp(X))

    # Compute enhanced magnitude spectra
    enh_est = y_bpg_specific*X.unsqueeze(3)
    enh_ideal = y_true*X

    # Numerical stability
    eps = 1e-12
    enh_est[enh_est < 0] = 0
    enh_ideal[enh_ideal < 0] = 0

    # Ensure enh_ideal matches the shape of enh_est
    enh_ideal = (enh_ideal.unsqueeze(3)).repeat(1, 1, 1, bpg_outputs.size()[2])

    # Loss for each bpg specific expert
    bpg_specific_loss = torch.sum(F.mse_loss(enh_est, enh_ideal, reduction='none'), 2)
    
    if avg_or_sum == "sum":
        #loss = torch.sum(loss_mask * torch.sum(bpg_outputs * bpg_specific_loss, 2))
        loss = -torch.sum(loss_mask * torch.log(torch.sum(bpg_outputs * torch.exp(-0.5*bpg_specific_loss), 2) + eps))
    elif avg_or_sum == "mean":
        #loss = torch.sum(loss_mask * torch.sum(bpg_outputs * bpg_specific_loss, 2))/torch.sum(loss_mask)
        loss = -torch.sum(loss_mask * torch.log(torch.sum(bpg_outputs * torch.exp(-0.5*bpg_specific_loss), 2) + eps))/torch.sum(loss_mask)

    return loss


def srmr_loss(X, torch_scaler, y_pred, y_true, loss_mask, device, avg_or_sum):
    """
    Calculate signal approximation loss, which is defined as
    the mean squared error between the T-F signal enhanced using
    the estimated mask and the signal enhanced using the ideal
    mask.

    Args:
        X (3D tensor): normalized features
        torch_scaler (TorchStandardScaler)
        y_pred (3D tensor): predicted mask
        y_true (3D tensor): ideal mask
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        device: whether the CPU or GPU
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: signal approximation loss
    """
    # Un-normalize features
    X = torch_scaler.inverse_transform(X)

    # Take features corresponding to current time window
    X = X[:, :, -y_true.size()[2]:]

    # Transform log power spectrum features back to magnitude spectrum
    # to calculate the loss
    X[loss_mask] = torch.sqrt(torch.exp(X[loss_mask]))

    # Compute enhanced magnitude spectra
    enh_est = y_pred*X
    enh_ideal = y_true*X

    # Numerical stability
    enh_est[enh_est < 0] = 0
    enh_ideal[enh_ideal < 0] = 0

    # Calculate SRMR
    enh_est = enh_est.permute(0, 2, 1)
    loss_mask = loss_mask.permute(0, 2, 1)
    srmr = objective_intelligibility.srmr.calculate_srmr(enh_est, 16000, loss_mask)
    losses = -srmr.to(device)
    
    #batch_size = X.size()[0]
    #losses = (torch.zeros(batch_size, 1, requires_grad=True)).to(device)
    #for batch in range(batch_size):
    #    curr = enh_est[batch, :, :][loss_mask[batch, :, :]].view(-1, X.size()[2]).to('cpu')
    #    curr = torch.transpose(curr, 0, 1)
    #    srmr = objective_intelligibility.srmr.calculate_srmr(curr, 16000)
    #    losses[batch] = -srmr.to(device)

    if avg_or_sum == "sum":
        loss = torch.sum(losses)
    elif avg_or_sum == "mean":
        loss = torch.mean(losses)

    return loss


def compress_sig_approx(X, torch_scaler, y_pred, y_true, p, loss_mask, avg_or_sum):
    """
    Calculate signal approximation loss for power compressed signals

    Args:
        X (3D tensor): normalized features
        torch_scaler (TorchStandardScaler)
        y_pred (3D tensor): predicted mask
        y_true (3D tensor): ideal mask
        p (float): power compression exponent
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: signal approximation loss
    """
    # Un-normalize features
    X = torch_scaler.inverse_transform(X)

    # Take features corresponding to current time window
    X = X[:, :, -y_true.size()[2]:]

    # Transform log power spectrum features back to magnitude spectrum
    # to calculate the loss
    X[loss_mask] = torch.sqrt(torch.exp(X[loss_mask]))

    # Compute enhanced magnitude spectra
    enh_est = y_pred*X
    enh_ideal = y_true*X

    # Numerical stability
    enh_est[enh_est < 0] = 0
    enh_ideal[enh_ideal < 0] = 0

    if avg_or_sum == "sum":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est**p, enh_ideal**p, reduction='none'))
    elif avg_or_sum == "mean":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est**p, enh_ideal**p, reduction='none'))/(torch.sum(loss_mask))

    return loss


def log_sig_approx(X, torch_scaler, y_pred, y_true, loss_mask, avg_or_sum):
    """
    Calculate signal approximation loss in log spectral domain.

    Args:
        X (3D tensor): normalized features
        torch_scaler (TorchStandardScaler)
        y_pred (3D tensor): predicted mask
        y_true (3D tensor): ideal mask
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: signal approximation loss
    """
    # Un-normalize features
    X = torch_scaler.inverse_transform(X)

    # Take features corresponding to current time window
    X = X[:, :, -y_true.size()[2]:]

    # Transform log power spectrum features back to magnitude spectrum
    # to calculate the loss
    X[loss_mask] = torch.sqrt(torch.exp(X[loss_mask]))

    # Compute enhanced magnitude spectra
    enh_est = y_pred*X
    enh_ideal = y_true*X

    # Numerical stability
    eps = 1e-12
    enh_est[enh_est < eps] = eps
    enh_ideal[enh_ideal < eps] = eps

    if avg_or_sum == "sum":
        loss = torch.sum(loss_mask * F.mse_loss(torch.log(enh_est), torch.log(enh_ideal), reduction='none'))
    elif avg_or_sum == "mean":
        loss = torch.sum(loss_mask * F.mse_loss(torch.log(enh_est), torch.log(enh_ideal), reduction='none'))/(torch.sum(loss_mask))

    return loss


def gamma_sig_approx(X, torch_scaler, y_pred, y_true, p, loss_mask, device, avg_or_sum):
    """
    Calculate signal approximation loss in gammatone domain

    Args:
        X (3D tensor): normalized features
        torch_scaler (TorchStandardScaler)
        y_pred (3D tensor): predicted mask
        y_true (3D tensor): ideal mask
        p (float): power compression exponent
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        device: whether CPU or GPU
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: signal approximation loss
    """
    # Un-normalize features
    X = torch_scaler.inverse_transform(X)

    # Take features corresponding to current time window
    X = X[:, :, -y_true.size()[2]:]

    # Transform log power spectrum features back to magnitude spectrum
    # to calculate the loss
    X[loss_mask] = torch.sqrt(torch.exp(X[loss_mask]))

    # Compute enhanced magnitude spectra
    enh_est = y_pred*X
    enh_ideal = y_true*X

    # Numerical stability
    enh_est[enh_est < 0] = 0
    enh_ideal[enh_ideal < 0] = 0

    # Gammatone filterbank
    nfft = 2*(X.size()[2]-1)
    fs = 16000
    n_filters = 22
    width = 1
    fmin = 50
    fmax = fs/2
    gfb, _ = fft_weights(nfft, fs, n_filters, width, fmin, fmax, int(nfft/2+1))
    gfb = (torch.from_numpy(np.float32(gfb))).to(device)
    enh_est = torch.matmul(enh_est, gfb.T)
    enh_ideal = torch.matmul(enh_ideal, gfb.T)

    # Reshape loss_mask
    loss_mask = loss_mask[:, :, 0:22]

    if avg_or_sum == "sum":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est**p, enh_ideal**p, reduction='none'))
    elif avg_or_sum == "mean":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est**p, enh_ideal**p, reduction='none'))/(torch.sum(loss_mask))

    return loss


def mel_sig_approx(X, torch_scaler, y_pred, y_true, loss_mask, device, avg_or_sum):
    """
    Calculate signal approximation loss in mel-frequency domain

    Args:
        X (3D tensor): normalized features
        torch_scaler (TorchStandardScaler)
        y_pred (3D tensor): predicted mask
        y_true (3D tensor): ideal mask
        loss_mask (3D tensor): relevant T-F bins over which to
        calculate the loss
        device: whether CPU or GPU
        avg_or_sum (str): whether to average or sum loss

    Returns:
        loss: signal approximation loss
    """
    # Un-normalize features
    X = torch_scaler.inverse_transform(X)

    # Take features corresponding to current time window
    X = X[:, :, -y_true.size()[2]:]

    # Transform log power spectrum features back to magnitude spectrum
    # to calculate the loss
    X[loss_mask] = torch.sqrt(torch.exp(X[loss_mask]))

    # Compute enhanced magnitude spectra
    enh_est = y_pred*X
    enh_ideal = y_true*X

    # Numerical stability
    enh_est[enh_est < 0] = 0
    enh_ideal[enh_ideal < 0] = 0

    # Mel filterbank
    mfb = librosa.filters.mel(sr=16000, n_fft=128, n_mels=22)
    mfb = (torch.from_numpy(mfb)).to(device)
    enh_est = torch.matmul(enh_est, mfb.T)
    enh_ideal = torch.matmul(enh_ideal, mfb.T)

    # Reshape loss_mask
    loss_mask = loss_mask[:, :, 0:22]

    if avg_or_sum == "sum":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est, enh_ideal, reduction='none'))
    elif avg_or_sum == "mean":
        loss = torch.sum(loss_mask * F.mse_loss(enh_est, enh_ideal, reduction='none'))/(torch.sum(loss_mask))

    return loss
