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
from preprocessing_Dorians_features import normalize_features # noqa401
from sklearn.model_selection import train_test_split
from basic_ml import basic_ml


names_features_targets_file = os.path.join(os.path.dirname(__file__),
                                           'datasets',
                                           'names_features_targets.txt')
with open(names_features_targets_file, 'rb') as f:
    names, features, targets = pickle.load(f)
augmented_features, augmented_targets = augmentate_dataset(features, targets)

normalized_augmented_features = normalize(augmented_features)
# an alternative approach to normalizing
# features = np.transpose(normalize_features(features))
unique_names = name_unique_features(names,
                                    normalized_augmented_features)

random_state = 0
# Other random states may be tried to check that similar results are achieved
random.seed(random_state)

# Models that will be used are chosen
ml_models = ['SVC', 'DT', 'KNN', 'RF', 'MPL', 'my_mlp']

# train and test sets are created
x_train, x_test, y_train, y_test = train_test_split(features, targets,
                                                    test_size=0.20,
                                                    random_state=random_state)
# test features are balanced
bal_x_test, bal_y_test = balance_dataset(x_test, y_test)
# and the repeated features are removed before presenting them to any model
# we will ensure that instances send to the models dont have repeated features
unique_bal_x_test = remove_notunique_features(unique_names, names, bal_x_test)
# testing data for all approaches is ready
unique_x_train = remove_notunique_features(unique_names, names, x_train)
# training data without changes ready
bal_x_train, bal_y_train = balance_dataset(x_train, y_train)
unique_bal_x_train = remove_notunique_features(unique_names, names, bal_x_train)
# balanced training data ready
aug_x_train, aug_y_train = augmentate_dataset(x_train, y_train)
unique_aug_x_train = remove_notunique_features(unique_names, names, aug_x_train)
# augmented training data ready


output_file = os.path.join(os.path.dirname(__file__),
                           'ml_results.csv')
with open(output_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Normal", "Balance data", "Augment data"])
    for ml_model in ml_models:
        acc_basic = basic_ml(unique_x_train, unique_bal_x_test,
                             y_train, bal_y_test,
                             ml_model, random_state=random_state)

        acc_bal = basic_ml(unique_bal_x_train, unique_bal_x_test,
                           bal_y_train, bal_y_test,
                           ml_model, random_state=random_state)

        acc_augmented = basic_ml(unique_aug_x_train, unique_bal_x_test,
                                 aug_y_train, bal_y_test,
                                 ml_model, random_state=random_state)

        round_accuracies = [round(acc, 2) for acc in [acc_basic,
                                                      acc_bal,
                                                      acc_augmented]]
        writer.writerow([ml_model] + round_accuracies)
