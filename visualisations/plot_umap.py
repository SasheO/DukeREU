import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    from importlib.metadata import *
except ImportError:  # Python < 3.10 (backport)
    from importlib_metadata import *
import umap.umap_ as umap
import json
import pickle

from get_averages_and_std_per_phoneme import get_phonemes_from_single_file, zmuv_normalize_phoneme_class

manner_of_articulation_map = {"vowels":{"iy","ih", 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'ax', 'ax-h', 'ix'},
                             "dipthongs":{'ey', 'aw', 'ay', 'oy', 'ow'},
                             "semi-vowels": {'l', 'el', 'r', 'w', 'y', 'er', 'axr'},
                             "stops": {'b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch'},
                             "fricatives": {'s', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'hh', 'hv'},
                             "nasals": {'m', 'em', 'n', 'nx', 'ng', 'eng', 'en'},
                             "silence": {'dx', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'h', 'pau', 'epi', 'q'},
                             "h#": {"h#"}}

vowels_and_consonants_map = {"vowels":{"iy","ih", 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'ax', 'ax-h', 'ix', 'ey', 'aw', 'ay', 'oy', 'ow'},
                             "semi-vowels": {'l', 'el', 'r', 'w', 'y', 'er', 'axr'},
                             "consonants": {'b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'hh', 'hv', 'm', 'em', 'n', 'nx', 'ng', 'eng', 'en', 'dx', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'h', 'pau', 'epi', 'q'},
                             "h#": {"h#"}}

def plot_umap(features, phonemes_from, colour_map=None, colour_map_name=None):
    
    '''
    colour_map (dict): maps label of phoneme classification groups to a set/list of phonemes in that group. for example, the manner of articulation colour_map (from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4100697) would be {"vowels":{"iy","ih", 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'ax', 'ax-h', 'ix'},                             "dipthongs":{'ey', 'aw', 'ay', 'oy', 'ow'},"semi-vowels": {'l', 'el', 'r', 'w', 'y', 'er', 'axr'},"stops": {'b', 'd', 'g', 'p', 't', 'k', 'jh', 'ch'},"fricatives": {'s', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'hh', 'hv'},"nasals": {'m', 'em', 'n', 'nx', 'ng', 'eng', 'en'},"silence": {'dx', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'h', 'pau', 'epi', 'q'},"h#": {"h#"}} 
    '''
    
    features = np.array(features[::20], dtype=float)
    features = np.nan_to_num(features, copy=True, posinf=0, neginf=0)
    features = features[:,:65]
    print(features.shape)
    phonemes_from = phonemes_from[::20]
    print(len(phonemes_from))
    # return
    # copy pasted most of this code from mnist example
    reducer = umap.UMAP(random_state=42, low_memory=True)
    embedding = reducer.fit_transform(features)

    # if issues, may be due to nan values. think how to deal with sparse data? maybe cut off at some point.
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = phonemes_from
    if len(colors) < 5000:
        s = 10
    elif len(colors) < 100000:
        s=3
    else:
        s =1
    
    if colour_map:
        number_to_phoneme = {}

        for y,x in phoneme_to_number.items():
            number_to_phoneme[x]=y
        phonemes_from_colour_coded = []
        for phoneme_num in phonemes_from:
            phoneme = number_to_phoneme[phoneme_num]
            added = False
            for indx in range(len(colour_map)):
                if phoneme in list(colour_map.items())[indx][1]:
                    phonemes_from_colour_coded.append(indx)
                    added = True
                    break
            if added:
                continue
            else:
                phonemes_from_colour_coded.append(len(colour_map)+1)
                print(phoneme)
        colors = phonemes_from_colour_coded

    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="Spectral", s=s)
    plt.setp(ax, xticks=[], yticks=[])
    plt.title("Phonemes", fontsize=18)

    if colour_map:
        try:
            plt.colorbar(boundaries=np.arange(len(set(colors)))-0.5).set_ticks(ticks=np.arange(len(set(colors))), labels=list(colour_map.keys())) #+["other"])
        except:
            m=plt.colorbar(boundaries=np.arange(len(set(colors)))-0.5)
            m.set_ticks(ticks=np.arange(len(set(colors)))) #+["other"])
            m.set_ticklabels(list(colour_map.keys()))
        if colour_map_name:
            plt.title("Phonemes according to "+ colour_map_name, fontsize=18)
    else:
        plt.colorbar().ax.tick_params(labelsize=10)

    output_image_num = 1
    output_image = str(output_image_num)+"_.png"
    while os.path.isfile(output_image):
        output_image_num += 1
        output_image = str(output_image_num)+"_.png"

    plt.savefig(output_image)





def get_features_from_files(files_searching, normalize=False, folder_with_mean_and_stds=None):
    '''
    folder_with_mean_and_stds (str): folder path containing files names (phoneme)_average.txt or (phoneme)_std.txt which can be loaded using np.loadtxt. Files should contain average and std phoneme values with features of length weighted_average_phoneme_length.
    '''
    
    
    if normalize:
        if folder_with_mean_and_stds:
            pass
        else:
            raise Exception("Must pass in folder_with_mean_and_stds as an argument if normalize is set to True")

    features = []
    phoneme_to_number = {}
    phonemes_from = [] # this will put the phonemes here which will be used to colour code the data

    current = 0 # used to sample how many files running through for testing
    
    for filename in files_searching:
        phonemes_features_dict = get_phonemes_from_single_file(filename)
        for phoneme in phonemes_features_dict:
            features_for_phoneme = phonemes_features_dict[phoneme]
            if normalize: # if looking for normalized features, normalized using the saved mean and std
                if phoneme == "sil": # sil is the librispeech equivalent of h# in timit
                    _phoneme = "h#"
                else:
                    _phoneme = phoneme
                phoneme_mean = np.loadtxt(folder_with_mean_and_stds+"/"+_phoneme+"_average.txt", dtype="float64")
                phoneme_std = np.loadtxt(folder_with_mean_and_stds+"/"+_phoneme+"_std.txt", dtype="float64")
                features_for_phoneme,_,__ = zmuv_normalize_phoneme_class(features_for_phoneme, False, phoneme_mean, phoneme_std)
            
            features.extend(features_for_phoneme)
            if phoneme in phoneme_to_number:
                pass
            else:
                phoneme_to_number[phoneme] = len(phoneme_to_number)+1
            phonemes_from.extend([phoneme_to_number[phoneme]]*len(features_for_phoneme))


        current += 1
        print(current)
        # if current == 5:
        #     break
    if normalize:
        with open('../saved_variables/features_from_files_normalized.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([features,phonemes_from,phoneme_to_number], f)
    else:
        with open('../saved_variables/features_from_files.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump([features,phonemes_from,phoneme_to_number], f)
    return features,phonemes_from,phoneme_to_number


if __name__=="__main__":
    f = open("../saved_variables/files_searching.txt", "r")
    files_searching = [file.strip() for file in f.readlines()] # list of direct paths to files searching
    # make path relative
    index_before = files_searching[0].index("Sentence")
    files_searching1 = [file.replace(file[:index_before],"../") for file in files_searching] # list of direct paths to files searching
    files_searching = files_searching1
    f.close()
    
    print(1)
    with open("../saved_variables/features_from_files.pkl", "rb") as f:  # Python 3: open(..., 'rb')
        features, phonemes_from, phoneme_to_number = pickle.load(f)
    print(1)
    
    '''
    features, phonemes_from, phoneme_to_number = get_features_from_files(files_searching)
    '''
    # plotting all classes
    plot_umap(features, phonemes_from) 

    # plotting with colour code by class
    print("moa")
    map_using = manner_of_articulation_map ### CHANGE
    name_of_plot = "Manner of Articulation" ### CHANGE
    plot_umap(features, phonemes_from, map_using, name_of_plot)

    # plotting with colour code by class
    print("vowels and consonants")
    map_using = vowels_and_consonants_map ### CHANGE
    name_of_plot = "Vowels and Consonants" ### CHANGE
    plot_umap(features, phonemes_from, map_using, name_of_plot)
    
    
    print(1)
    with open("../saved_variables/features_from_files_normalized.pkl", "rb") as f:  # Python 3: open(..., 'rb')
        features, phonemes_from, phoneme_to_number = pickle.load(f)
    print(1)
    '''

    folder_with_mean_and_stds="../saved_variables/average_and_std_phoneme_feature/"
    features, phonemes_from, phoneme_to_number  = get_features_from_files(files_searching, normalize=True, folder_with_mean_and_stds=folder_with_mean_and_stds)
    '''

    # plotting all classes
    plot_umap(features, phonemes_from) 

    # plotting with colour code by class
    print("moa")
    map_using = manner_of_articulation_map ### CHANGE
    name_of_plot = "Manner of Articulation" ### CHANGE
    plot_umap(features, phonemes_from, map_using, name_of_plot)

    # plotting with colour code by class
    print("vowels and consonants")
    map_using = vowels_and_consonants_map ### CHANGE
    name_of_plot = "Vowels and Consonants" ### CHANGE
    plot_umap(features, phonemes_from, map_using, name_of_plot)
