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
import yaml
import importlib.util
from config.ml_models import ml_models
from config.ml_models import classifiers
from config.ml_models import dataset_types
from config.hyperparameters_grid import grid
from sklearn.model_selection import GridSearchCV


def write_yaml_to_file(py_obj, filename):
    with open(f'{filename}.yaml', 'w',) as f:
        yaml.dump(py_obj, f, sort_keys=False)
    print('Written to file successfully')


def k_folds_ml(x_train, y_train, model, random_state=0):
    """
    Train the desired model.

    The hyperparameters of the models are chosen using 5-fold cross validation.
    """
    current_classifier = classifiers[model]
    current_grid = grid[model]
    rf_cv = GridSearchCV(estimator=current_classifier(),
                         param_grid=current_grid,
                         cv=5)
    rf_cv.fit(x_train, y_train)
    return rf_cv.best_params_


test_balanced_dataset_file = os.path.join(os.path.dirname(__file__),
                                          'datasets', 'test',
                                          'balanced_test_dataset.txt')
with open(test_balanced_dataset_file, 'rb') as g:
    balanced_x_test, balanced_y_test = pickle.load(g)

test_normal_dataset_file = os.path.join(os.path.dirname(__file__),
                                        'datasets', 'test',
                                        'normal_test_dataset.txt')
with open(test_normal_dataset_file, 'rb') as g:
    normal_x_test, normal_y_test = pickle.load(g)

output_file_balanced = os.path.join(os.path.dirname(__file__),
                                    'ml_results_k_fold_tested_in_balanced.csv')
with open(output_file_balanced, 'w') as f_balanced:
    writer_balanced = csv.writer(f_balanced)
    writer_balanced.writerow(["Name"] + dataset_types)
    output_file_normal = os.path.join(os.path.dirname(__file__),
                                      'ml_results_k_fold_tested_in_normal.csv')
    with open(output_file_normal, 'w') as f_normal:
        writer_normal = csv.writer(f_normal)
        writer_normal.writerow(["Name"] + dataset_types)
        for ml_model in ml_models:
            print(f"Model: {ml_model}")
            acc_balanced = dict()
            acc_normal = dict()
            for method in dataset_types:
                this_dataset_file = os.path.join(os.path.dirname(__file__),
                                                 'datasets', 'train',
                                                 f'{method}_train_dataset.txt')
                with open(this_dataset_file, 'rb') as f:
                    method_x_train, method_y_train = pickle.load(f)
                hyperparams = k_folds_ml(method_x_train, method_y_train,
                                         model=ml_model)
                write_yaml_to_file(hyperparams,
                                   os.path.join(os.path.dirname(__file__),
                                                'config', 'hyperparams',
                                                f'{method}_{ml_model}'))
                current_classifier = classifiers[ml_model]
                clf = current_classifier(**hyperparams)
                clf.fit(method_x_train, method_y_train)
                acc_balanced[method] = clf.score(balanced_x_test,
                                                 balanced_y_test)
                acc_normal[method] = clf.score(normal_x_test, normal_y_test)
                method_file = os.path.join(os.path.dirname(__file__),
                                           'config', 'models',
                                           f'{method}_trained_model.txt')
                with open(method_file, 'wb') as f_method:
                    pickle.dump(clf, f_method)
            round_accuracies_balanced = [round(acc, 2)
                                         for acc in [acc_balanced[method_here]
                                         for method_here in dataset_types]]
            round_accuracies_normal = [round(acc, 2)
                                       for acc in [acc_normal[method_here]
                                       for method_here in dataset_types]]
            writer_balanced.writerow([ml_model] + round_accuracies_balanced)
            writer_normal.writerow([ml_model] + round_accuracies_normal)
