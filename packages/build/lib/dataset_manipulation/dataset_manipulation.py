"""Exploit symmetries in polynomials to augmentate or balance the dataset."""
import numpy as np
import math
import random
from .exploit_symmetries import give_all_symmetries
# from sklearn.preprocessing import normalize

nvar = 3


def augmentate_dataset(features, targets, timings):
    """
    Multiply the size of the dataset by 6.

    Arguments:
    features: list(list(numpy.float))
    targets: list(numpy.float)
    """
    symmetric_features = []
    symmetric_targets = []
    symmetric_timings = []
    for features, target, timing in zip(features, targets, timings):
        symmetric_features += give_all_symmetries(features, int(target))
        symmetric_targets += list(range(math.factorial(nvar)))
        symmetric_timings += list(timing)
    return np.array(symmetric_features), np.array(symmetric_targets), \
        np.array(symmetric_timings)


def balance_dataset(features, targets, timings):
    """
    Balance the dataset so all targets are almost equally common.

    Arguments:
    features: list(list(numpy.float))
    targets: list(numpy.float)
    """
    balanced_features = []
    balanced_targets = []
    balanced_timings = []
    for features, target, timing in zip(features, targets, timings):
        symmetric_features = give_all_symmetries(features, int(target))
        possible_targets = list(range(math.factorial(nvar)))
        new_target = random.choice(possible_targets)
        balanced_features.append(symmetric_features[new_target])
        balanced_targets.append(new_target)
        balanced_timings.append(timing[new_target])
    return np.array(balanced_features), np.array(balanced_targets),\
        np.array(balanced_timings)


def name_unique_features(names, features):
    """
    Return the name of unique features.

    When two features share the same value for all the instances
    one of them is not considered unique.
    """
    new_features = []
    new_names = []
    rep = 0
    for index, feature in enumerate(zip(*features)):
        if (any([np.array_equal(feature, ex_feature)
                 for ex_feature in new_features])
                or np.std(feature) == 0):
            rep += 1
        elif feature.count(feature[0]) == len(feature):
            print(names[index])
        else:
            # if 'max_in_polys_max_sig'==names[index][:20]:
            #     print("Check ", feature.count(feature[0])==len(feature))
            #     print(names[index])
            # print(len(feature))
            new_features.append(feature)
            new_names.append(names[index])
    return new_names


def get_unique_feature_names(unique_names, names, features):
    """Return the features corresponding to a name in 'unique_names'."""
    unique_features = []
    for index, feature in enumerate(zip(*features)):
        if names[index] in unique_names:
            unique_features.append(feature)
    return np.transpose(unique_features)


def remove_notunique_features(names, features):
    # creating some targets and timing because the function requires them
    targets = [0]*len(features)
    timings = [[0, 0]]*len(features)
    augmented_features, _, _ = augmentate_dataset(features, targets, timings)
    # normalized_augmented_features = normalize(augmented_features)
    unique_names = name_unique_features(names, augmented_features)
    unique_features = []
    for index, feature in enumerate(zip(*features)):
        if names[index] in unique_names:
            unique_features.append(feature)
    return unique_names, np.transpose(unique_features)
