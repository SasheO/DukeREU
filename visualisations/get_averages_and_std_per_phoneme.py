import numpy as np

def zmuv_normalize_phoneme_class(feature_matrix, calculate_mean_and_std=True, phoneme_mean=None, phoneme_std=None):
    '''
    return the normalized feature class of matrix phoneme_class_features_matrix, the mean and standard deviations
    inputs:
        feature_matrix (np.array): a 2d feature matrix of various instances of a phoneme. The phonemes instances are stacked on the z (?) axis
        calculate_mean_and_std (bool): whether to calculate mean and std or not. should be false if inputting values for phoneme_mean and phoneme_std
        phoneme_mean (np.array):  the average feature
        phoneme_std (np.array):  the standard deviation of features
        
    returns:
        normalized_features (np.array): the normalized feature matrix 
        phoneme_mean (np.array):  the average feature
        phoneme_std (np.array):  the standard deviation of features
    '''
    if calculate_mean_and_std:
        phoneme_mean, phoneme_std = np.mean(feature_matrix,0),np.std(feature_matrix,0)
    normalized_features = (feature_matrix - phoneme_mean)/(phoneme_std)
    
    return normalized_features, phoneme_mean, phoneme_std

def get_instances_of_phoneme_from_single_file(input_filename_or_file_phoneme_features_dictionary, phoneme):  ## TODO: this is a misnomer and is confusing bc it can also get instances of phonemes from a dictionary and not only a file. RENAME!
    '''
    this returns a numpy matrix of features of a phoneme class
    inputs:
        input_filename_or_file_phoneme_features_dictionary (str or dict): the input file or a dictionary output from get_phonemes_from_single_file
        phoneme (str): the phoneme class (assuming working with timit database)
        
        returns:
        list of numpy array features of a phoneme class OR empty list if phoneme does not occur in file at all
    '''
    if type(input_filename_or_file_phoneme_features_dictionary)==str:
        phoneme_features_dict = get_phonemes_from_single_file(input_filename_or_file_phoneme_features_dictionary)
    else:
        phoneme_features_dict = input_filename_or_file_phoneme_features_dictionary
        
    if phoneme in phoneme_features_dict:
        return phoneme_features_dict[phoneme]
    else:
        return []
    
def get_phonemes_from_single_file(filename):
    '''
    this returns a dicitonary mapping phoneme names to numpy 1-D matrices of features of the phoneme class of a particular length (padded with nans if data is less than) found in a file
    inputs:
        filename (str): the input file
        
    '''
    output_dictionary = {}
    f = open(filename, "r")
    lines = f.readlines()
    
    f.close()
    for line in lines: 
        line = line.split()
        current_phoneme = line[-1]
        line = line[:-1]
        line = np.array(line,dtype="float64")
        if current_phoneme in output_dictionary:
            output_dictionary[current_phoneme].append(line)
        else:
            output_dictionary[current_phoneme] = [line]
        
    return output_dictionary

if __name__=="__main__":
    f = open("../saved_variables/files_searching.txt", "r")
    files_searching = [file.strip() for file in f.readlines()] # list of direct paths to files searching
    f.close()
    # make path relative
    index_before = files_searching[0].index("Sentence")
    files_searching1 = [file.replace(file[:index_before],"../") for file in files_searching] # list of direct paths to files searching
    files_searching = files_searching1
    

    # TODO: EDIT THIS TO MAKE MORE EFFICIENT TO RUN ON WHOLE DATASET
    # get averages and stds using the weighted total average length
    output_folder = "../saved_variables/average_and_std_phoneme_feature/" ### CHANGE THIS
    phonemes_to_features = {}
    count = 0
    for file in files_searching:
        phonemes_from_file = get_phonemes_from_single_file(file)
        for phoneme in phonemes_from_file:
            if phoneme in phonemes_to_features:
                pass
            else:
                phonemes_to_features[phoneme]=[]
            phonemes_to_features[phoneme].extend(get_instances_of_phoneme_from_single_file(phonemes_from_file, phoneme))
        count += 1
        print("file #"+str(count))
        # if count == 5:
        #     break
            
    for phoneme in phonemes_to_features:
        feature_matrix = np.array(phonemes_to_features[phoneme], dtype="float64")
        feature_matrix_normed, phoneme_mean, phoneme_std = zmuv_normalize_phoneme_class(feature_matrix, True)
        np.savetxt(output_folder+phoneme+"_average.txt", phoneme_mean)
        np.savetxt(output_folder+phoneme+"_std.txt", phoneme_std)

    