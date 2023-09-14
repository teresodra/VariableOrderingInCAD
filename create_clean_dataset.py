"""This file contains a function that given the raw dataset containing
the sets of polynomials and its timings for each order, creates a dataset
containing a set of unique features and its class"""

import pickle
import numpy as np
from replicating_Dorians_features import extract_features
import importlib
if isinstance(importlib.util.find_spec('dataset_manipulation'), type(None)):
    from dataset_manipulation import remove_notunique_features
else:
    from packages.dataset_manipulation import remove_notunique_features
from from_poly_set_to_features import poly_set_feature_extractor
from find_filename import find_dataset_filename
from find_filename import find_other_filename


def create_dataframe(dataset):
    all_features = []
    all_labels = dataset[1][:]
    all_timings = dataset[2][:]
    all_original_polynomials = []
    for index, all_projections in enumerate(dataset[0]):
        original_polynomials = all_projections[0][0]
        all_original_polynomials.append(original_polynomials)
    names, all_features =\
        poly_set_feature_extractor(all_original_polynomials,
                                   determine_standarization=True,
                                   determine_unique_features=True)
    return np.array(all_original_polynomials), np.array(names),\
        np.array(all_features), np.array(all_labels), np.array(all_timings)


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
    unique_features_filename = find_other_filename("unique_features")
    with open(unique_features_filename, 'wb') as unique_features_file:
        pickle.dump(clean_dataset['names'], unique_features_file)
    # Some timings are expressed as "Over 30", this is changed here
    clean_dataset['timings'] = \
        np.array([[convert_to_timing(timings_ordering)
                  for timings_ordering in timings_problem]
                  for timings_problem in my_dataset['timings']])
    for key in my_dataset:
        if key not in clean_dataset:
            clean_dataset[key] = my_dataset[key]
    print("CLEAN", clean_dataset.keys())
    with open(clean_dataset_filename, 'wb') as clean_dataset_file:
        pickle.dump(clean_dataset, clean_dataset_file)

# dataset_filename = os.path.join(os.path.dirname(__file__),
#                                 'DatasetsBeforeProcessing',
#                                 'dataset_without_repetition_return_ncells.txt')
# clean_dataset_filename = os.path.join(os.path.dirname(__file__),
#                                       'datasets',
#                                       'clean_dataset.txt')
# cleaning_dataset(dataset_filename, clean_dataset_filename)


def convert_to_timing(timing_str):
    if timing_str == "Over 30":
        return 60
    if timing_str == "Over 60":
        return 120
    return float(timing_str)
