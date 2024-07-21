import numpy as np
from sklearn import preprocessing


def get_label_list(label_type):
    """ Get list of phonetic labels.

    Args:
        label_type (str): type of label to encode

    Returns:
        label_list (list): list of phonetic labels
    """
    # Select file containing label list
    if label_type == "phone":
        label_file = "phones/phone_list.txt"
    elif label_type == "phoneme":
        label_file = "phones/phoneme_list_arpabet.txt"
    elif label_type == "moa":
        label_file = "phones/moa_list.txt"
    elif label_type == "bpg":
        label_file = "phones/bpg_list.txt"
    elif label_type == "vuv":
        label_file = "phones/vuv_list.txt"
    elif label_type == "moa_vuv":
        label_file = "phones/moavuv_list.txt"

    with open(label_file, 'r') as f:
        label_list = f.readlines()
        for i in range(len(label_list)):
            label_list[i] = label_list[i].replace("\n", "")

    return label_list


def get_seq_list(labels, context_dependency):
    """ Get list of possible phone sequences

    Args:
        labels (list): list of possible labels
        context_dependency (str): amount of context to model

    Returns:
        seq (list): list of possible sequences 
    """

    seq = []

    # Biphones: previous phone-current phone
    if context_dependency == "bi":
        for i in range(len(labels)):
            for j in range(len(labels)):
                seq.append(labels[i]+"-"+labels[j])

    # Triphones: previous phone-current phone-following phone
    elif context_dependncy == "tri":
        for i in range(len(labels)):
            for j in range(len(labels)):
                for k in range(len(labels)):
                    seq.append(labels[i]+"-"+labels[j]+"-"+labels[k])

    # Monophone for beginning and end phones
    for label in labels:
        seq.append(label)

    return seq


def get_label_encoder(label_type, context_dependency="mono"):
    """ Get label encoder

    Args:
        label_type (str): type of label to encode (phone or phoneme)

    Returns:
        le (preprocessing.LabelEncoder): label encoder

    """
    # Get list of labels
    labels = get_label_list(label_type)
    
    # List of possible sequences
    if context_dependency != "mono":
        labels = get_seq_list(labels, context_dependency)

    le = preprocessing.LabelEncoder()
    le.fit(labels)

    return le


def map_phones(phones, conf_dict):
    if conf_dict["phonemap"] == "none":
        mapped = phones
    else:
        # Open
        file_obj = open(conf_dict["phonemap"], "r")
        x = file_obj.readlines()
        file_obj.close()

        # Creates a dictionary where keys are phonemes and values of moa
        phone_dict = {}
        for i in range(0, len(x)):
            temp = x[i].split()
            phone_dict[temp[0]] = temp[1]

            # If phonemes
            if conf_dict["bpg"] == "phoneme":
                if conf_dict["num_phonemes"] == 39:
                    phone_dict[temp[0]] = temp[2]

        # Map phones to reduced sets
        mapped = []
        for phone in phones:
            mapped.append(phone_dict[phone])
    
    return mapped

    
def classes_to_sequences(classes, context_dependency, le):
    """
    Convert phone classes to sequences (currently supports biphones
    and triphones)

    Args:
        classes (list): phone classes (phones, phonemes, moas, etc.)
        context_dependency (str): amount of phonetic context
        le (LabelEncoder): object for encoding strs as ints

    Returns:
        seq (list): sequence of phone classes
    """

    # Transform list of classes into array of ints
    encoded_classes = le.transform(classes)

    # Find indices where phone class changes
    trans_idx = np.argwhere(np.diff(encoded_classes)!=0)+1
    trans_idx = trans_idx.reshape((len(trans_idx),))
    trans_idx = np.concatenate((np.array([0]), trans_idx))

    # List for storing sequences of phone classes 
    seq = []

    # Generate context-dependent phones
    # Biphone models: previous phone-current phone
    # Triphone models: previous phone-current phone-following phone
    # First and last phones are treated as monophones
    for i in range(len(trans_idx)):
        if i == 0:
            seq.append(np.tile(np.array(classes[trans_idx[i]]), (trans_idx[i+1]-trans_idx[i],)))
        elif i > 0 and i < len(trans_idx)-1:
            if context_dependency == "bi":
                seq.append(np.tile(np.array(classes[trans_idx[i-1]] + "-" + classes[trans_idx[i]]), (trans_idx[i+1]-trans_idx[i],)))
            elif context_dependency == "tri":
                seq.append(np.tile(np.array(classes[trans_idx[i-1]] + "-" + classes[trans_idx[i]] + "-" + classes[trans_idx[i+1]]), (trans_idx[i+1]-trans_idx[i],)))
        else:
            seq.append(np.tile(np.array(classes[trans_idx[i]]), (len(classes)-trans_idx[i],)))

    seq = list(np.hstack(seq))

    return seq
