"""Exploit symmetries in polynomials to augmentate or balance the dataset."""
import numpy as np
import math
import random
from .exploit_symmetries import give_all_symmetries

nvar = 3


def augmentate_dataset(features, targets):
    """
    Multiply the size of the dataset by 6.

    Arguments:
    features: list(list(numpy.float))
    targets: list(numpy.float)
    """
    symmetric_features = []
    symmetric_targets = []
    for features, target in zip(features, targets):
        symmetric_features += give_all_symmetries(features, int(target))
        symmetric_targets += list(range(math.factorial(nvar)))
    return np.array(symmetric_features), np.array(symmetric_targets)


def balance_dataset(features, targets):
    """
    Balance the dataset so all targets are almost equally common.

    Arguments:
    features: list(list(numpy.float))
    targets: list(numpy.float)
    """
    balanced_features = []
    balanced_targets = []
    for features, target in zip(features, targets):
        symmetric_features = give_all_symmetries(features, int(target))
        possible_targets = list(range(math.factorial(nvar)))
        new_target = random.choice(possible_targets)
        balanced_features.append(symmetric_features[new_target])
        balanced_targets.append(new_target)
    return np.array(balanced_features), np.array(balanced_targets)


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
        else:
            new_features.append(feature)
            new_names.append(names[index])
    return new_names


def remove_notunique_features(unique_names, names, features):
    """Return the features corresponding to a name in 'unique_names'."""
    unique_features = []
    for index, feature in enumerate(zip(*features)):
        if names[index] in unique_names:
            unique_features.append(feature)
    return np.transpose(unique_features)
