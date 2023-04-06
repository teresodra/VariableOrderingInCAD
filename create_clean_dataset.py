import pickle
import numpy as np
from replicating_Dorians_features import extract_features
import importlib
if isinstance(importlib.util.find_spec('dataset_manipulation'), type(None)):
    from dataset_manipulation import remove_notunique_features
else:
    from packages.dataset_manipulation import remove_notunique_features


def cleaning_dataset(dataset_filename, clean_dataset_filename):
    with open(dataset_filename, 'rb') as f:
        dataset = pickle.load(f)
    original_polys_list, names, features_list, targets_list, timings_list = extract_features(dataset)

    # working with raw features
    features = np.array(features_list)
    unique_names, unique_features = remove_notunique_features(names, features)

    targets = np.array(targets_list)
    timings = np.array(timings_list)
    original_polys = np.array(original_polys_list)

    with open(clean_dataset_filename, 'wb') as clean_dataset_file:
        dataset = pickle.dump((original_polys, unique_names,
                               unique_features, targets, timings),
                              clean_dataset_file)


# dataset_filename = os.path.join(os.path.dirname(__file__),
#                                 'DatasetsBeforeProcessing',
#                                 'dataset_without_repetition_return_ncells.txt')
# clean_dataset_filename = os.path.join(os.path.dirname(__file__),
#                                       'datasets',
#                                       'clean_dataset.txt')
# cleaning_dataset(dataset_filename, clean_dataset_filename)
