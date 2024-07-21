from evaluate_model_transform import test


if __name__ == '__main__':
    # Inputs
    save_dir = "/media/lab/Seagate Expansion Drive/Sashe/"
    conf_file = "conf/fft_mask_dp_fft/LSTMModelTransform.txt"
    model_name = "model2" # should be same as most recent model from train.py
    test_sets = ["test_hint_office_0_1_3", "test_hint_lecture_0_1_3", "test_hint_stairway_0_1_3_90"] #, "test_hint_office_0_1_3", "test_lecture_1_1_3"] # "test_cuny_aula_carolina_0_1_4_90_3", "test_cuny_aula_carolina_1_1_4_90_3" 

    for test_set in test_sets:
        test(conf_file, model_name, test_set, save_dir)
