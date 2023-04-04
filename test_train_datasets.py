"""
The experiments in [1] are replicated with some changes.

The first change is that the testing data is balanced, so that all targets
are almost equally common.
Then we use three training sets; dataset as in [1], balanced dataset
and data augmentation dataset.

[1]Florescu, D., England, M. (2020). A Machine Learning Based Software Pipeline
to Pick the Variable Ordering for Algorithms with Polynomial Inputs.
Bigatti, A., Carette, J., Davenport, J., Joswig, M., de Wolff, T. (eds)
Mathematical Software, ICMS 2020. ICMS 2020. Lecture Notes in Computer Science,
vol 12097. Springer, Cham. https://doi.org/10.1007/978-3-030-52200-1_30
"""


import os
import pickle
import random
import csv
import yaml
import importlib.util
# Check if 'dataset_manipulation' is installed
if isinstance(importlib.util.find_spec('dataset_manipulation'), type(None)):
    from dataset_manipulation import name_unique_features
    from dataset_manipulation import remove_notunique_features
    from dataset_manipulation import balance_dataset
    from dataset_manipulation import augmentate_dataset
else:
    from packages.dataset_manipulation import name_unique_features
    from packages.dataset_manipulation import remove_notunique_features
    from packages.dataset_manipulation import balance_dataset
    from packages.dataset_manipulation import augmentate_dataset
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


def count_instances(my_dataset, instance):
    return sum(my_dataset==instance)


names_features_targets_file = os.path.join(os.path.dirname(__file__),
                                           'datasets',
                                           'clean_dataset.txt')
with open(names_features_targets_file, 'rb') as f:
    original_polys, names, features, targets, timings = pickle.load(f)

augmented_features, augmented_targets, augmented_timings = augmentate_dataset(features, targets, timings)

normalized_augmented_features = normalize(augmented_features)
unique_names = name_unique_features(names,
                                    augmented_features)

random_state = 0

x = dict() # to keep the features
y = dict() # to keep the labels
t = dict() # to keep the timings
# train and test sets are created
not_unique_x_normal_train, not_unique_x_normal_test, y['train_normal'], y['test_normal'], t['train_normal'], t['test_normal'] = train_test_split(features, targets, timings,
                                                                                           test_size=0.20,
                                                                                           random_state=random_state)

not_unique_balanced_x_test, y['test_balanced'], t['test_balanced'] = balance_dataset(not_unique_x_normal_test, y['test_normal'], t['test_normal'])
x['test_balanced'] = remove_notunique_features(unique_names, names, not_unique_balanced_x_test)
# testing data for all approaches is ready
# all tests will be done in balanced but the others are also computed
not_unique_augmented_x_test, y['test_augmented'], t['test_augmented'] = augmentate_dataset(not_unique_x_normal_test, y['test_normal'], t['test_normal'])
x['test_augmented'] = remove_notunique_features(unique_names, names, not_unique_augmented_x_test)
x['test_normal'] = remove_notunique_features(unique_names, names, not_unique_x_normal_test)

x['train_normal'] = remove_notunique_features(unique_names, names, not_unique_x_normal_train)
# normal training data ready
not_unique_balanced_x_train, y['train_balanced'], t['train_balanced'] = balance_dataset(not_unique_x_normal_train, y['train_normal'], t['train_normal'])
x['train_balanced'] = remove_notunique_features(unique_names, names, not_unique_balanced_x_train)
# balanced training data ready
not_unique_augmented_x_train, y['train_augmented'], t['train_augmented'] = augmentate_dataset(not_unique_x_normal_train, y['train_normal'], t['train_normal'])
x['train_augmented'] = remove_notunique_features(unique_names, names, not_unique_augmented_x_train)
# augmented training data ready


dataset_info_file = os.path.join(os.path.dirname(__file__),
                                 'datasets',
                                 'dataset_instances.csv')
with open(dataset_info_file, 'w') as f_dataset_info:
    writer = csv.writer(f_dataset_info)
    writer.writerow(['dataset'] + ['zero','one','two','three','four','five','total'])
    for usage in ['train', 'test']:
        for method in ['normal', 'balanced', 'augmented']:
            print(f"y['{usage}_{method}'])", len(y[f'{usage}_{method}']))
            this_dataset_file = os.path.join(os.path.dirname(__file__),
                                            'datasets', usage,
                                            f'{method}_{usage}_dataset.txt')
            with open(this_dataset_file, 'wb') as f:
                pickle.dump((x[f'{usage}_{method}'], y[f'{usage}_{method}']), f)
                
            writer.writerow([f'{usage} {method} dataset']
                            + [str(count_instances(y[f'{usage}_{method}'], i))
                               for i in range(6)]
                            + [str(len(y[f'{usage}_{method}']))])