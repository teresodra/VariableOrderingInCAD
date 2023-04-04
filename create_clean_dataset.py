import pickle
import numpy as np
from replicating_Dorians_features import extract_features
from basic_ml import use_tf, basic_ml
from itertools import product
import sys
import os
import csv


dataset_file = os.path.join(os.path.dirname(__file__), 'DatasetsBeforeProcessing', 'dataset_without_repetition_return_ncells.txt')
f = open(dataset_file, 'rb')
dataset = pickle.load(f)
original_polys_list, names, features_list, targets_list, timings_list = extract_features(dataset)

# working with raw features
features = np.array(features_list)
targets = np.array(targets_list)
timings = np.array(timings_list)
original_polys = np.array(original_polys_list)

clean_dataset_file = os.path.join(os.path.dirname(__file__),
                                  'datasets',
                                  'clean_dataset.txt')
g = open(clean_dataset_file, 'wb')
dataset = pickle.dump((original_polys, names, features, targets, timings), g)
