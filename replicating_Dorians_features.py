"""
IS THIS BEING USED?
YES, IT IS!
"""

import itertools
# from xml.sax.handler import all_features
import numpy as np


def aveg(given_list):
    return sum(given_list)/len(given_list)


def aveg_not_zero(given_list):
    return sum(given_list)/max(1, len([1 for elem in given_list
                                      if elem != 0]))


def identity(input):
    return input


def sign(input):
    if type(input) == list:
        return [sign(elem) for elem in input]
    else:
        if input > 0:
            return 1
        elif input < 0:
            return -1
        elif input == 0:
            return 0
        else:
            raise Exception("How is this possible?")


def create_features(degrees, variable=0, sv=False,
                    operations=[sum, max, aveg, aveg_not_zero]):
    sign_or_not = [identity, sign]
    features = []
    features_names = []
    for choice in itertools.product(operations,
                                    sign_or_not,
                                    operations,
                                    sign_or_not):
        feature_description = (choice[0].__name__
                               + "sign" * (choice[1].__name__ == "sign")
                               + "_in_polys_" + choice[2].__name__ + "_"
                               + "sign" * (choice[3].__name__ == "sign")
                               + "of_" + "sum_of_" * sv + "degrees_of_var_"
                               + str(variable) + "_in_monomials")
        feature_value = \
            choice[0](choice[1]([choice[2](choice[3](degrees_in_poly))
                                 for degrees_in_poly in degrees]))
        features.append(feature_value)
        features_names.append(feature_description)
    return features, features_names


def extract_features(dataset):
    my_dataset = dict()
    all_features = []
    all_labels = []
    all_timings = []
    all_original_polynomials = []
    all_projections = []
    all_cells = []
    for index, projections in enumerate(dataset[0]):
        all_projections.append(projections)
        original_polynomials = projections[0][0]
        # the original polynomials are the initial polynomials of any
        # of the possible projections (also of the first one)
        all_original_polynomials.append(original_polynomials)
        all_labels.append(dataset[1][index])
        all_timings.append(dataset[2][index])
        all_cells.append(dataset[3][index])
        names, instance_features = \
            features_from_set_of_polys(
                original_polynomials,
                operations=[sum, max, aveg, aveg_not_zero])
        all_features.append(instance_features)
    my_dataset['polynomials'] = np.array(all_original_polynomials)
    my_dataset['names'] = np.array(names)
    my_dataset['features'] = np.array(all_features)
    my_dataset['labels'] = np.array(all_labels)
    my_dataset['timings'] = np.array(all_timings)
    my_dataset['projections'] = np.array(all_projections)
    my_dataset['cells'] = np.array(all_cells)
    return my_dataset


def features_from_set_of_polys(original_polynomials,
                               operations=[sum, max, aveg, aveg_not_zero]):
    instance_features = []
    names = []
    nvar = len(original_polynomials[0][0]) - 1
    for var in range(nvar):
        var_features, var_names = \
            compute_features_for_var(original_polynomials,
                                     var,
                                     operations=operations)
        instance_features += var_features
        names += var_names
    return names, instance_features


def compute_features_for_var(original_polynomials, var,
                             operations=[sum, max, aveg]):
    '''Given polynomials and a variable computes the features'''
    degrees = [[monomial[var] for monomial in poly]
               for poly in original_polynomials]
    var_features, var_features_names = \
        create_features(degrees,
                        variable=var,
                        operations=operations)
    sdegrees = \
        [[sum(monomial) for monomial in poly if monomial[var] != 0] + [0]
         for poly in original_polynomials]
    svar_features, svar_features_names = \
        create_features(sdegrees,
                        variable=var,
                        sv=True,
                        operations=operations)
    var_names = var_features_names + svar_features_names
    var_features = var_features + svar_features
    return var_features, var_names
