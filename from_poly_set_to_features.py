"""This file will contain the functions necessary to convert
a list of sets of polynomials to a list of their features.
This features will be unique and standarised"""
import numpy as np
import pickle
from packages.dataset_manipulation import augmentate_dataset
from find_filename import find_other_filename
from replicating_Dorians_features import features_from_set_of_polys


def poly_set_feature_extractor(sets_of_polys, determine_unique_features=False,
                               determine_standarization=False):
    """Given a list of polynomial sets will return a list of its features"""
    features_list = []
    for set_of_polys in sets_of_polys:
        names, features = features_from_set_of_polys(set_of_polys)
        features_list.append(features)
    if determine_unique_features:
        # if we want to find unique feature names
        find_unique_features(names, features_list)
    unique_names, unique_features = get_unique_features(names, features_list)
    if determine_standarization:
        find_standarizing_values(unique_names, unique_features)
    standarized_features = get_standarized_features(unique_names, unique_features)
    return names, standarized_features


# def features_set_of_polys(original_polynomials):
#     instance_features = []
#     names = []
#     nvar = len(original_polynomials[0][0]) - 1
#     for var in range(nvar):
#         degrees = [[monomial[var] for monomial in poly]
#                    for poly in original_polynomials]
#         var_features, var_features_names = create_features(degrees,
#                                                            variable=var)
#         instance_features += var_features
#         names += var_features_names
#         sdegrees = [[sum(monomial) for monomial in poly
#                      if monomial[var]!=0]+[0]
#                      for poly in original_polynomials]
#         svar_features, svar_features_names = create_features(sdegrees,
#                                                              variable=var,
#                                                              sv=True)
#         instance_features += svar_features
#         names += svar_features_names
#     return names, instance_features


def find_unique_features(names, features):
    """
    Saves the name of unique features in the assigned file.

    When two features share the same value for all the instances,
    or they are the same after adition or multiplication,
    one of them is not considered unique.
    """
    # we want to look for uniqueness after augmenting to discard
    # some that might look equal
    # creating labels and timing for the augmentate_dataset function
    labels = [0]*len(features)
    timings = [[0, 0]]*len(features)
    augmented_features, _, _ = augmentate_dataset(features, labels, timings)
    # now we look for the unique features
    unique_features = []
    unique_names = []
    for index, feature in enumerate(zip(*augmented_features)):
        if (any([np.array_equal(feature, ex_feature)
                 for ex_feature in unique_features])
                or np.std(feature) == 0):
            # check if this feature has been already recorded
            pass
        elif feature.count(feature[0]) == len(feature):
            # check if it is a constant list
            pass
        else:
            # if none of the previous conditions then
            unique_features.append(feature)
            unique_names.append(names[index])
    unique_names_filename = find_other_filename('unique_names')
    with open(unique_names_filename, 'wb') as unique_names_file:
        pickle.dump(unique_names, unique_names_file)


def get_unique_features(names, features):
    """Return the features corresponding to a name in 'unique_names'."""
    # We recover the list of unique feature names
    unique_names_filename = find_other_filename('unique_names')
    with open(unique_names_filename, 'rb') as unique_names_file:
        unique_names = pickle.load(unique_names_file)
    # we keep only the features that are unique
    unique_features = []
    index = 0
    for feature in zip(*features):
        if names[index] in unique_names:
            unique_features.append(feature)
        index += 1
    return unique_names, np.transpose(unique_features)


def find_standarizing_values(names, features_list):
    """Finds and saves the mean and std of the different features
    so that features can be standarised in a consistent way
    before giving them to the machine learning models"""
    standarizing_values = dict()
    for name, features in zip(names, features_list):
        standarizing_values[name] = (np.mean(features), np.std(features))
    standarizing_values_filename = find_other_filename('standarizing_values')
    with open(standarizing_values_filename, 'wb') as standarizing_values_file:
        pickle.dump(standarizing_values, standarizing_values_file)


def get_standarized_features(names, features):
    """Returns the standarised features."""
    # We recover the list of unique feature names
    standarizing_values_filename = find_other_filename('standarizing_values')
    with open(standarizing_values_filename, 'rb') as standarizing_values_file:
        standarizing_values = pickle.load(standarizing_values_file)
    # we keep only the features that are unique
    standarized_features = []
    index = 0
    for index, feature in enumerate(zip(*features)):
        mean, std = standarizing_values[names[index]]
        standarized_features.append((feature-mean)/std)
    return np.transpose(standarized_features)
