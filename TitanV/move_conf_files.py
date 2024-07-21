from shutil import copyfile

conf_file = "conf/ratio_mask_dp_fft/taslp2022_nonideal_phonemes/LSTM_1layer_but_rev_log_fft_batch16_phoneme_sigapprox.txt"
model_dir = "/media/batcave/personal/chu.kevin/TitanV/MaskEstimationPytorch/exp/ratio_mask_dp_fft/taslp2022_nonideal_phonemes/LSTM_1layer_but_rev_log_fft_batch16_phoneme_sigapprox/model1"
model_dir2 = "exp/ratio_mask_dp_fft/taslp2022_nonideal_phonemes/LSTM_1layer_but_rev_log_fft_batch16_phoneme_sigapprox/model1"

phoneme_list_file = "phones/phoneme_list_arpabet.txt"

with open(phoneme_list_file, "r") as f:
    phonemes = f.readlines()

for phoneme in phonemes:
    phoneme = phoneme.replace("\n", "")
    
    bpg_specific_model_dir = model_dir + "/" + phoneme
    copyfile(conf_file, bpg_specific_model_dir + "/conf.txt")

    bpg_specific_model_dir2 = model_dir2 + "/" + phoneme
    copyfile(conf_file, bpg_specific_model_dir2 + "/conf.txt")
