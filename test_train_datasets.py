import numpy as np
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
from config.general_values import purposes
from config.general_values import dataset_qualities
from sklearn.model_selection import train_test_split
from find_filename import find_dataset_filename
from find_filename import find_other_filename
from math import log
from sklearn.model_selection import GroupShuffleSplit 


def count_instances(my_dataset, instance):
    return sum(my_dataset == instance)


def create_train_test_datasets():
    clean_dataset_filename = find_dataset_filename('clean')
    with open(clean_dataset_filename, 'rb') as clean_dataset_file:
        dataset = pickle.load(clean_dataset_file)

    ###
    # Instead of creating dictionaries for features, labels,...abs
    # maybe it's better to create a dictionary for each dataset:
    # train/test, normal/balanced/augmented
    ###
    datasets = dict()
    for purpose in purposes:
        for quality in dataset_qualities:
            datasets[purpose + '_' + quality] = dict()
    # train and test sets are created
    random_state = 0
    print(dataset.keys())
    train_inds, test_inds = my_train_test_split(dataset['subdir'])
    for key in dataset:
        if key != 'names':
            datasets['Train_Biased'][key] = \
                [dataset[key][i] for i in train_inds]
            datasets['Test_Biased'][key] = \
                [dataset[key][i] for i in test_inds]

    # datasets['Train_Biased']['features'], \
    #     datasets['Test_Biased']['features'], \
    #     datasets['Train_Biased']['labels'], \
    #     datasets['Test_Biased']['labels'], \
    #     datasets['Train_Biased']['timings'], \
    #     datasets['Test_Biased']['timings'], \
    #     datasets['Train_Biased']['projections'], \
    #     datasets['Test_Biased']['projections'], \
    #     datasets['Train_Biased']['cells'], \
    #     datasets['Test_Biased']['cells'] = \
        # train_test_split(
        #                  dataset['features'],
        #                  dataset['labels'],
        #                  dataset['timings'],
        #                  dataset['projections'],
        #                  dataset['cells'],
        #                  test_size=0.20,
        #                  random_state=random_state)
    keys = ['features', 'timings', 'cells']
    for purpose in purposes:
        datasets[f'{purpose}_Balanced'] = \
            {key: elem for key,
             elem in zip(keys, balance_dataset(
                                   *[datasets[f'{purpose}_Biased'][key2]
                                     for key2 in keys], nvar=3)) ##CHOOSE NVAR WELL
             }
        datasets[f'{purpose}_Balanced']['labels'] = \
            [timings.index(min(timings)) for timings in datasets[f'{purpose}_Balanced']['timings']]
        datasets[f'{purpose}_Augmented'] = \
            {key: elem for key,
             elem in zip(keys, augmentate_dataset(
                                   *[datasets[f'{purpose}_Biased'][key2]
                                     for key2 in keys], nvar=3))
             }
        print(f"features in {purpose}_Augmented", len(datasets[f'{purpose}_Augmented']['features'][0]))
        datasets[f'{purpose}_Augmented']['labels'] = \
            [timings.index(min(timings)) for timings in datasets[f'{purpose}_Augmented']['timings']]
    for purpose in purposes:
        for quality in dataset_qualities:
            this_dataset_filename = \
                find_dataset_filename(purpose, dataset_quality=quality)
            with open(this_dataset_filename, 'wb') as this_dataset_file:
                pickle.dump(datasets[purpose + '_' + quality],
                            this_dataset_file)

    # The following code is to count how many instances of each are there in the different datasets
    # Sould be substitute by another function

        # {datasets[f'{purpose}_balanced'][key]: elem for elem in balance_dataset(datasets[f'{purpose}_balanced'][key2] for key2 in keys) for key in keys}
        # x[f'{purpose}_augmented'], y[f'{purpose}_augmented'], t[f'{purpose}_augmented'] = augmentate_dataset(x[f'{purpose}_normal'], y[f'{purpose}_normal'], t[f'{purpose}_normal'])
#     dataset_info_file = find_dataset_filename('instances')
#     with open(dataset_info_file, 'w') as f_dataset_info:
#         writer = csv.writer(f_dataset_info)
#         writer.writerow(['dataset'] + ['zero', 'one', 'two', 'three', 'four', 'five', 'total'])
#         for purpose in purposes:
#             for method in ['normal', 'balanced', 'augmented']:
#                 this_dataset_filename = find_dataset_filename(purpose, method=method)
#                 with open(this_dataset_filename, 'wb') as this_dataset_file:
#                     if method == 'normal':
#                         pickle.dump((x[f'{purpose}_{method}'], y[f'{purpose}_{method}'], t[f'{purpose}_{method}'], p[f'{purpose}_{method}']), this_dataset_file)
#                     else:
#                         pickle.dump((x[f'{purpose}_{method}'], y[f'{purpose}_{method}'], t[f'{purpose}_{method}']), this_dataset_file)

#                 writer.writerow([f'{purpose} {method} dataset']
#                                 + [str(count_instances(y[f'{purpose}_{method}'], i))
#                                 for i in range(6)]
#                                 + [str(len(y[f'{purpose}_{method}']))])


def create_regression_datasets(dataset_quality='Augmented',
                               taking_logarithms=True):
    for purpose in purposes:
        existing_dataset_filename = find_dataset_filename(
                                        purpose,
                                        dataset_quality=dataset_quality)
        # we will use the augmented dataset here
        with open(existing_dataset_filename, 'rb') as this_dataset_file:
            regression_dataset = pickle.load(this_dataset_file)
            regression_dataset['labels'] = \
                [timings[0] for timings
                 in regression_dataset['timings']]
            if taking_logarithms:
                regression_dataset['labels'] = \
                    [log(label) for label
                     in regression_dataset['labels']]
            new_dataset_filename = find_dataset_filename(
                                       purpose,
                                       dataset_quality=dataset_quality,
                                       paradigm='Regression')
            with open(new_dataset_filename, 'wb') as this_dataset_file:
                pickle.dump(regression_dataset, this_dataset_file)
            # classification_dataset = regression_dataset
            # classification_dataset['labels'] = \
            #     [np.argmin(timings) for timings
            #      in regression_dataset['timings']]


def my_train_test_split(groups):
    splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
    split = splitter.split(groups, groups=groups)
    train_inds, test_inds = next(split)
    return train_inds, test_inds


# create_regression_datasets(taking_logarithms=False)

# create_train_test_datasets()
