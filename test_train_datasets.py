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
from find_filename import find_other_filename
from math import log


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
    x = dict()  # to keep the features
    y = dict()  # to keep the labels
    t = dict()  # to keep the timings
    p = dict()  # to keep the projections
    c = dict()  # to keep the number of cells
    # train and test sets are created
    random_state = 0
    x['train_normal'], x['test_normal'], \
        y['train_normal'], y['test_normal'], \
        t['train_normal'], t['test_normal'], \
        p['train_normal'], p['test_normal'] = \
        train_test_split(dataset['features'],
                         dataset['targets'],
                         dataset['timings'],
                         dataset['projections'],
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
                this_dataset_filename = find_dataset_filename(usage, method=method)
                with open(this_dataset_filename, 'wb') as this_dataset_file:
                    if method == 'normal':
                        pickle.dump((x[f'{usage}_{method}'], y[f'{usage}_{method}'], t[f'{usage}_{method}'], p[f'{usage}_{method}']), this_dataset_file)
                    else:
                        pickle.dump((x[f'{usage}_{method}'], y[f'{usage}_{method}'], t[f'{usage}_{method}']), this_dataset_file)

                writer.writerow([f'{usage} {method} dataset']
                                + [str(count_instances(y[f'{usage}_{method}'], i))
                                for i in range(6)]
                                + [str(len(y[f'{usage}_{method}']))])


def create_regression_datasets(taking_logarithms=True):
    for usage in ['train', 'test']:
        this_dataset_filename = find_dataset_filename(usage,
                                                      method='augmented')
        # we will use the augmented dataset here
        with open(this_dataset_filename, 'rb') as this_dataset_file:
            X, Y, T = pickle.load(this_dataset_file)
            if taking_logarithms:
                Y = [log(timings[0]) for timings in T]
            else:
                Y = [timings[0] for timings in T]
            this_dataset_filename =\
                find_dataset_filename(usage, method='regression')
            with open(this_dataset_filename, 'wb') as this_dataset_file:
                pickle.dump((X, Y, T), this_dataset_file)
