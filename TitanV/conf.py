import re

def read_conf(conf_file):
    """ Read configuration file as dict

    Args:
        conf_file (str): configuration file

    Returns:
        conf_dict (dict): configuration file as dict

    """
    with open(conf_file, "r") as file_obj:
        conf = file_obj.readlines()

    conf = list(map(lambda x: x.replace("\n", ""), conf))

    # Convert conf to dict
    conf_dict = {}
    for line in conf:
        if "=" in line:
            contents = line.split(" = ")
            conf_dict[contents[0]] = convert_string(contents[0], contents[1])

    conf_dict["num_features"] = (1 + int(conf_dict["deltas"]) + int(conf_dict["deltaDeltas"])) * \
                                (conf_dict["num_coeffs"] + int(conf_dict["use_energy"]))

    # Splicing, if applicable
    if "window_size" in conf_dict.keys():
        conf_dict["num_features"] = conf_dict["window_size"] * conf_dict["num_features"]

    return conf_dict


def convert_string(key, value):
    """ Convert string into appropriate data type based on the
    dictionary key

    Args:
        key (str): dictionary key
        value (str): value expressed as a string

    Returns:
        converted_value (varies): value converted into appropriate data type

    """

    try:
        # Ints
        if "num" in key or "size" in key or "len" in key:
            converted_value = int(value)
        # Floats
        else:
            converted_value = float(value)
    except ValueError:
        # Tuple
        if re.match("\(\d*,\d*\)", value):
            temp = re.sub("\(|\)", "", value).split(",")
            converted_value = (int(temp[0]), int(temp[1]))
        # Boolean
        elif value == "True" or value == "False":
            converted_value = value == "True"
        # String
        else:
            converted_value = value

    return converted_value
