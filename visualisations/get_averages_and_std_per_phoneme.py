import numpy as np
import json

def zmuv_normalize_phoneme_class(phoneme_class_features_matrix, calculate_mean_and_std=False, phoneme_mean=None, phoneme_std=None):
    '''
    return the normalized feature class of matrix phoneme_class_features_matrix, the mean and standard deviations
    inputs:
        phoneme_class_features_matrix (np.array): a 3d feature matrix of various instances of a phoneme. The phonemes instances are stacked on the z (?) axis
    returns:
        normalized_features (np.array): the normalized feature matrix 
        phoneme_mean (np.array):  the average feature (when nan values are ignored)
        phoneme_std (np.array):  the standard deviation (when nan values are ignored)
    '''
    if calculate_mean_and_std:
        phoneme_mean, phoneme_std = np.nanmean(phoneme_class_features_matrix,0),np.nanstd(phoneme_class_features_matrix,0)
    normalized_features = (phoneme_class_features_matrix - phoneme_mean)/(phoneme_std)
    
    return normalized_features, phoneme_mean, phoneme_std

def get_instances_of_phoneme_from_single_file(input_filename_or_file_phoneme_features_dictionary, phoneme, intended_phoneme_feature_matrix_length=None):
    '''
    this returns a numpy matrix of features of a phoneme class of a particular length (padded with nans if data is less than)
    inputs:
        filename (str): the input file
        phoneme (str): the phoneme class (assuming working with timit database)
        intended_phoneme_feature_matrix_length (int): the matrix length intended, such as a weighted average of the phoneme class lengths. if the features are not up to this length, it will be padded
        
        returns:
        numpy matrix of features of a phoneme class of a particular length (padded with nans if data is less than)
        OR 
        none if phoneme does not occur in file at all
    '''
    if type(input_filename_or_file_phoneme_features_dictionary)==str:
        
        if intended_phoneme_feature_matrix_length==None:
            raise Exception("If you are inputting a file to this function, you must also give a 'intended_phoneme_feature_matrix_length' value.")
        phoneme_features_dict = get_phonemes_from_single_file(input_filename_or_file_phoneme_features_dictionary, intended_phoneme_feature_matrix_length)
    else:
        phoneme_features_dict = input_filename_or_file_phoneme_features_dictionary
        
    if phoneme in phoneme_features_dict:
        return phoneme_features_dict[phoneme]
    else:
        return
    
def get_phonemes_from_single_file(filename, intended_phoneme_feature_matrix_length):
    '''
    this returns a dicitonary mapping phoneme names to numpy matrices of features of the phoneme class of a particular length (padded with nans if data is less than) found in a file
    inputs:
        filename (str): the input file
        intended_phoneme_feature_matrix_length (int): the matrix length intended, such as a weighted average of the phoneme class lengths. if the features are not up to this length, it will be padded
    '''
    output_dictionary = {}
    previous_phoneme = ""
    f = open(filename, "r")
    lines = f.readlines()
    for line in lines: 
        line = line.split()
        current_phoneme = line[-1]
        line = line[:-1]
        
        if current_phoneme == previous_phoneme:
            if len(output_dictionary[current_phoneme][-1])<intended_phoneme_feature_matrix_length:
                output_dictionary[current_phoneme][-1].append(line)
        else:
            if previous_phoneme:
                a = np.empty((len(output_dictionary[previous_phoneme][-1][-1])))
                a[:] = np.nan
                while len(output_dictionary[previous_phoneme][-1])<intended_phoneme_feature_matrix_length:
                    output_dictionary[previous_phoneme][-1].append(a)
            if current_phoneme in output_dictionary:
                output_dictionary[current_phoneme].append([line])
            else:
                output_dictionary[current_phoneme] = [[line]]
        previous_phoneme = current_phoneme
    if output_dictionary:
        a = np.empty((len(output_dictionary[previous_phoneme][-1][-1])))
        a[:] = np.nan
        while len(output_dictionary[previous_phoneme][-1])<intended_phoneme_feature_matrix_length:
            output_dictionary[previous_phoneme][-1].append(a)
    
    for phoneme in output_dictionary:
        output_dictionary[phoneme] = np.array(output_dictionary[phoneme],dtype="float")
    return output_dictionary

if __name__=="__main__":
    
    
    with open('../saved_variables/average_phoneme_lengths.json') as json_file:
        average_phoneme_lengths = json.load(json_file)
    weighted_average_phoneme_length = int(average_phoneme_lengths['total'])
    f = open("../saved_variables/files_searching.txt", "r")
    files_searching = [file.strip() for file in f.readlines()] # list of direct paths to files searching
    # make path relative
    index_before = files_searching[0].index("Sentence")
    files_searching1 = [file.replace(file[:index_before],"../") for file in files_searching] # list of direct paths to files searching
    files_searching = files_searching1
    f.close()


    output_folder = "../average_and_std_phoneme_feature/weighted_average_total/"
    phonemes_to_features = {}

    # get averages and stds using the weighted total average length
    count = 0
    for file in files_searching:
        phonemes_from_file = get_phonemes_from_single_file(file, weighted_average_phoneme_length)
        for phoneme in phonemes_from_file:
            if phoneme in phonemes_to_features:
                
                pass
            else:
                phonemes_to_features[phoneme]=[]
            phonemes_to_features[phoneme].extend(get_instances_of_phoneme_from_single_file(phonemes_from_file, phoneme))
        count += 1
        print(count)
        # if count == 5:
        #     break
            
    for phoneme in phonemes_to_features:
        print(phoneme)
        feature_matrix = np.array(phonemes_to_features[phoneme], dtype="float64")
        feature_matrix_normed, phoneme_mean, phoneme_std = zmuv_normalize_phoneme_class(feature_matrix, True)
        np.savetxt(output_folder+phoneme+"_average.txt", phoneme_mean)
        np.savetxt(output_folder+phoneme+"_std.txt", phoneme_std)

    