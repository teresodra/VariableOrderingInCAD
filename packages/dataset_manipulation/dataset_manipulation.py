"""Exploit symmetries in polynomials to augmentate or balance the dataset."""
import numpy as np
import math
import random
from itertools import permutations
# from sklearn.preprocessing import normalize

nvar = 3


def augmentate_instance(features, timings, cells, nvar):
    variables = list(range(nvar))
    split_features = [features[i*len(features)//nvar:(i+1)*len(features)//nvar]
                      for i in range(nvar)]
    dict_timings = {str(perm): timing for perm, timing
                    in zip(permutations(variables), timings)}
    dict_cells = {str(perm): cell for perm, cell in zip(permutations(variables), cells)}
    augmented_features, augmented_timings, augmented_cells = [], [], []
    for perm in permutations(variables):
        augmented_features.append([feature for i in perm
                                  for feature in split_features[i]])
        augmented_timings.append([dict_timings[str(double_perm)]
                                  for double_perm in permutations(perm)])
        augmented_cells.append([dict_cells[str(double_perm)]
                                for double_perm in permutations(perm)])
    return augmented_features, augmented_timings, augmented_cells


def augmentate_dataset(all_features, all_timings, all_cells, nvar):
    """
    Multiply the size of the dataset by math.factorial(nvar).

    Arguments:
    features: list(list(numpy.float))
    targets: list(numpy.float)
    """
    augmented_features = []
    augmented_timings = []
    augmented_cells = []
    for features, timings, cells in \
            zip(all_features, all_timings, all_cells):
        new_features, new_timings, new_cells = \
            augmentate_instance(features, timings, cells, nvar)
        augmented_features += new_features
        augmented_timings += new_timings
        augmented_cells += new_cells
    return augmented_features, augmented_timings, augmented_cells


def balance_dataset(all_features, all_timings, all_cells, nvar):
    """
    Balance the dataset so all targets are almost equally common.

    Arguments:
    features: list(list(numpy.float))
    targets: list(numpy.float)
    """
    balanced_features = []
    balanced_timings = []
    balanced_cells = []
    for features, timings, cells in \
            zip(all_features, all_timings, all_cells):
        new_target = random.choice(list(range(math.factorial(nvar))))
        new_features, new_timings, new_cells = \
            augmentate_instance(features, timings, cells, nvar)
        balanced_features.append(new_features[new_target])
        balanced_timings.append(new_timings[new_target])
        balanced_cells.append(new_cells[new_target])
    return balanced_features, balanced_timings, balanced_cells

# features = [1,2,3,4,5,6]
# timings = [10,20,30,40,50,60]
# cells = [21,32,43,54,65,76]
# print(balance_dataset([features], [timings], [cells], 3))


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
        # print(feature)
        # if any([type(xfeature) == str for xfeature in feature]):
        # print(feature)
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


def remove_notunique_features(names, features, nvar=3):
    # creating some targets and timing because the function requires them
    timings = [list(range(math.factorial(nvar)))]*len(features)
    cells = [list(range(math.factorial(nvar)))]*len(features)
    augmented_features, _, _ = augmentate_dataset(features, timings, cells, nvar)
    # normalized_augmented_features = normalize(augmented_features)
    unique_names = name_unique_features(names, augmented_features)
    unique_features = []
    for index, feature in enumerate(zip(*features)):
        if names[index] in unique_names:
            unique_features.append(feature)
    return unique_names, np.transpose(unique_features)
