"""This file contains a function that given the raw dataset containing
the sets of polynomials and its timings for each order, creates a dataset
containing a set of unique features and its class"""

import re
import pickle
import numpy as np
from replicating_Dorians_features import extract_features
import importlib
if isinstance(importlib.util.find_spec('dataset_manipulation'), type(None)):
    from dataset_manipulation import remove_notunique_features
else:
    from packages.dataset_manipulation import remove_notunique_features
from find_filename import find_dataset_filename
from find_filename import find_other_filename


# def create_dataframe(dataset):
#     all_features = []
#     all_labels = dataset[1][:]
#     all_timings = dataset[2][:]
#     all_original_polynomials = []
#     for index, all_projections in enumerate(dataset[0]):
#         original_polynomials = all_projections[0][0]
#         all_original_polynomials.append(original_polynomials)
#     names, all_features =\
#         poly_set_feature_extractor(all_original_polynomials,
#                                    determine_standarization=True,
#                                    determine_unique_features=True)
#     return np.array(all_original_polynomials), np.array(names),\
#         np.array(all_features), np.array(all_labels), np.array(all_timings)


# dataset_filename = os.path.join(os.path.dirname(__file__),
#                                 'DatasetsBeforeProcessing',
#                                 'dataset_without_repetition_return_ncells.txt')
# with open(dataset_filename, 'rb') as f:
#     dataset = pickle.load(f)
# original_polys_list, names, features_list, labels_list, timings_list =\
#     create_dataframe(dataset)


def cleaning_dataset():
    dataset_filename = find_dataset_filename('unclean')
    clean_dataset_filename = find_dataset_filename('clean')
    with open(dataset_filename, 'rb') as f:
        dataset = pickle.load(f)
    my_dataset = extract_features(dataset)
    clean_dataset = dict()
    # # working with raw features
    # features = np.array(features_list)
    clean_dataset['names'], clean_dataset['features'] = \
        remove_notunique_features(my_dataset['names'],
                                  my_dataset['features'])
    print("features in biased", len(my_dataset['features'][0]))
    unique_features_filename = find_other_filename("unique_features")
    with open(unique_features_filename, 'wb') as unique_features_file:
        pickle.dump(clean_dataset['names'], unique_features_file)
    # Some timings are expressed as "Over 30", this is changed here
    clean_dataset['timings'] = \
        np.array([[convert_to_timing(timings_ordering)
                  for timings_ordering in timings_problem]
                  for timings_problem in my_dataset['timings']])
    # Some cells are expressed as "Over 30", this is changed here
    clean_dataset['cells'] = \
        np.array([convert_to_cells(cells_problem)
                  for cells_problem in my_dataset['cells']])
    for key in my_dataset:
        if key not in clean_dataset:
            clean_dataset[key] = my_dataset[key]
    with open(clean_dataset_filename, 'wb') as clean_dataset_file:
        pickle.dump(clean_dataset, clean_dataset_file)

# dataset_filename = os.path.join(os.path.dirname(__file__),
#                                 'DatasetsBeforeProcessing',
#                                 'dataset_without_repetition_return_ncells.txt')
# clean_dataset_filename = os.path.join(os.path.dirname(__file__),
#                                       'datasets',
#                                       'clean_dataset.txt')
# cleaning_dataset(dataset_filename, clean_dataset_filename)


def convert_to_timing(timing_str, penalization=2):
    if not contains_float(timing_str):
        return penalization * float(timing_str[5:])
    return float(timing_str)


def convert_to_cells(cells, penalization=2):
    int_cells = [int(cell) if contains_int(cell) else cell
                 for cell in cells]
    max_cells = max([cell for cell in int_cells if type(cell) == int])
    penalization_cells = [cell if type(cell) == int
                          else penalization*max_cells
                          for cell in int_cells]
    return penalization_cells


def contains_float(input_str):
    float_pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'
    match = re.search(float_pattern, input_str)
    return match is not None


def contains_int(input_str):
    int_pattern = r'^[-+]?\d+$'
    match = re.match(int_pattern, input_str)
    return match is not None

# cleaning_dataset()
