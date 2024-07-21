from evaluate import test


if __name__ == '__main__':
    # Inputs
    save_dir = "/media/batcave/personal/chu.kevin/TitanV/MaskEstimationPytorch"
    conf_file = "conf/ratio_mask_dp_fft/taslp2022_nonideal_phonemes/LSTM_1layer_but_rev_log_fft_889Hz_preemph_8kutts_batch16_phoneme_sigapprox.txt"
    model_name = "model1"
    bpg_model_name = "librispeech_rev_arpabet"
    test_sets = ["test_cuny_office_0_1_3",
                 "test_cuny_office_1_1_3",
                 "test_cuny_stairway_0_1_3_90",
                 "test_cuny_stairway_1_1_3_90",
                 "test_cuny_lecture_0_1_3",
                 "test_cuny_lecture_1_1_3"]
    bpg_labels = ["known",
                  "predicted_soft"]

    for test_set in test_sets:
        for bpg_label in bpg_labels:
            test(conf_file, model_name, test_set, bpg_label, bpg_model_name, save_dir)
