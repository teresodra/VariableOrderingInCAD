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
import csv
import importlib.util
# Check if 'dataset_manipulation' is installed
if isinstance(importlib.util.find_spec('dataset_manipulation'), type(None)):
    from dataset_manipulation import remove_notunique_features
    from dataset_manipulation import balance_dataset
    from dataset_manipulation import augmentate_dataset
else:
    from packages.dataset_manipulation import remove_notunique_features
    from packages.dataset_manipulation import balance_dataset
    from packages.dataset_manipulation import augmentate_dataset
from sklearn.model_selection import train_test_split
from find_filename import find_dataset_filename


def count_instances(my_dataset, instance):
    return sum(my_dataset == instance)


def create_train_test_datasets():
    clean_dataset_filename = find_dataset_filename('clean')
    with open(clean_dataset_filename, 'rb') as clean_dataset_file:
        _, names, features, targets, timings = pickle.load(clean_dataset_file)
    unique_names, unique_features = remove_notunique_features(names, features)

    x = dict()  # to keep the features
    y = dict()  # to keep the labels
    t = dict()  # to keep the timings
    # train and test sets are created
    random_state = 0
    x['train_normal'], x['test_normal'], y['train_normal'], y['test_normal'], t['train_normal'], t['test_normal'] = train_test_split(unique_features, targets, timings,
                                                                                                                                    test_size=0.20,
                                                                                                                                    random_state=random_state)
    for purpose in ['train', 'test']:
        x[f'{purpose}_balanced'], y[f'{purpose}_balanced'], t[f'{purpose}_balanced'] = balance_dataset(x[f'{purpose}_normal'], y[f'{purpose}_normal'], t[f'{purpose}_normal'])
        x[f'{purpose}_augmented'], y[f'{purpose}_augmented'], t[f'{purpose}_augmented'] = augmentate_dataset(x[f'{purpose}_normal'], y[f'{purpose}_normal'], t[f'{purpose}_normal'])
    dataset_info_file = find_dataset_filename('instances')
    with open(dataset_info_file, 'w') as f_dataset_info:
        writer = csv.writer(f_dataset_info)
        writer.writerow(['dataset'] + ['zero', 'one', 'two', 'three', 'four', 'five', 'total'])
        for usage in ['train', 'test']:
            for method in ['normal', 'balanced', 'augmented']:
                this_dataset_file = os.path.join(os.path.dirname(__file__),
                                                 'datasets', usage,
                                                 f'{method}_{usage}_dataset.txt')
                with open(this_dataset_file, 'wb') as f:
                    pickle.dump((x[f'{usage}_{method}'], y[f'{usage}_{method}']), f)

                writer.writerow([f'{usage} {method} dataset']
                                + [str(count_instances(y[f'{usage}_{method}'], i))
                                for i in range(6)]
                                + [str(len(y[f'{usage}_{method}']))])


# create_train_test_datasets()
