import os
import pickle
import csv
from config.ml_models import classifiers
from config.ml_models import all_models
from config.general_values import dataset_qualities
from config.hyperparameters_grid import grid
from sklearn.model_selection import GridSearchCV
from yaml_tools import write_yaml_to_file
from find_filename import find_dataset_filename
from find_filename import find_hyperparams_filename


def k_folds_ml(x_train, y_train, model, random_state=0):
    """
    Train the desired model.

    The hyperparameters of the models are chosen using 5-fold cross validation.
    """
    current_classifier = all_models[model]
    current_grid = grid[model]
    rf_cv = GridSearchCV(estimator=current_classifier(),
                         param_grid=current_grid,
                         cv=5,
                         verbose=10  # to get updates
                         )
    rf_cv.fit(x_train, y_train)
    return rf_cv.best_params_


def choose_hyperparams(model_name, paradigm, training_quality):
    """Given a ml_model and a method, a file with the hyperparameters
    chosen by cross validation is created"""
    this_dataset_file = find_dataset_filename('Train', dataset_quality=training_quality)
    with open(this_dataset_file, 'rb') as f:
        dataset = pickle.load(f)
    hyperparams = k_folds_ml(dataset['features'], dataset['labels'], model=model_name)
    print(hyperparams)
    hyperparams_filename = find_hyperparams_filename(model_name, paradigm, training_quality)
    print('new hyperparams_filename', hyperparams_filename)
    write_yaml_to_file(hyperparams, hyperparams_filename)


# test_balanced_dataset_file = os.path.join(os.path.dirname(__file__),
#                                           'datasets', 'test',
#                                           'balanced_test_dataset.txt')
# with open(test_balanced_dataset_file, 'rb') as g:
#     balanced_x_test, balanced_y_test = pickle.load(g)

# test_normal_dataset_file = os.path.join(os.path.dirname(__file__),
#                                         'datasets', 'test',
#                                         'normal_test_dataset.txt')
# with open(test_normal_dataset_file, 'rb') as g:
#     normal_x_test, normal_y_test = pickle.load(g)

# output_file_balanced = os.path.join(os.path.dirname(__file__),
#                                     'ml_results_k_fold_tested_in_balanced.csv')
# with open(output_file_balanced, 'w') as f_balanced:
#     writer_balanced = csv.writer(f_balanced)
#     writer_balanced.writerow(["Name"] + dataset_qualities)
#     output_file_normal = os.path.join(os.path.dirname(__file__),
#                                       'ml_results_k_fold_tested_in_normal.csv')
#     with open(output_file_normal, 'w') as f_normal:
#         writer_normal = csv.writer(f_normal)
#         writer_normal.writerow(["Name"] + dataset_qualities)
#         for ml_model in classifiers:
#             print(f"Model: {ml_model}")
#             acc_balanced = dict()
#             acc_normal = dict()
#             for method in dataset_qualities:
#                 this_dataset_file = os.path.join(os.path.dirname(__file__),
#                                                  'datasets', 'train',
#                                                  f'{method}_train_dataset.txt')
#                 with open(this_dataset_file, 'rb') as f:
#                     x_train, y_train, _ = pickle.load(f)
#                 hyperparams = k_folds_ml(x_train, y_train,
#                                          model=ml_model)
#                 write_yaml_to_file(hyperparams,
#                                    os.path.join(os.path.dirname(__file__),
#                                                 'config', 'hyperparams',
#                                                 f'{method}_{ml_model}'))
#                 current_classifier = all_models[ml_model]
#                 clf = current_classifier(**hyperparams)
#                 clf.fit(x_train, y_train)
#                 acc_balanced[method] = clf.score(balanced_x_test,
#                                                  balanced_y_test)
#                 acc_normal[method] = clf.score(normal_x_test, normal_y_test)
#                 method_filename = os.path.join(os.path.dirname(__file__),
#                                                'config', 'models',
#                                                f'{method}_trained_model.txt')
#                 with open(method_filename, 'wb') as method_file:
#                     pickle.dump(clf, method_file)
#             round_accuracies_balanced = [round(acc, 2)
#                                          for acc in [acc_balanced[method_here]
#                                          for method_here in dataset_qualities]]
#             round_accuracies_normal = [round(acc, 2)
#                                        for acc in [acc_normal[method_here]
#                                        for method_here in dataset_qualities]]
#             writer_balanced.writerow([ml_model] + round_accuracies_balanced)
#             writer_normal.writerow([ml_model] + round_accuracies_normal)


# model = 'KNN'
# method = 'balanced'
# choose_hyperparams(model, method)