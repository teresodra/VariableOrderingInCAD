import csv
import math
import pickle
import importlib.util
import numpy as np
from sklearn import metrics
from config.general_values import dataset_qualities
from config.ml_models import all_models
from config.ml_models import regressors
from config.ml_models import classifiers
from config.ml_models import heuristics
from find_filename import find_output_filename
from find_filename import find_dataset_filename
from find_filename import find_model_filename
from main_heuristics import ordering_choices_heuristics
# from train_models import ordering_choice_reinforcement
# from train_models import training_instances_reinforcement
# Check if 'dataset_manipulation' is installed
if isinstance(importlib.util.find_spec('dataset_manipulation'), type(None)):
    from dataset_manipulation import augmentate_instance
else:
    from packages.dataset_manipulation.dataset_manipulation import augmentate_instance


# def test_model(trained_model_filename, test_dataset_filename):
#     with open(trained_model_filename, 'rb') as trained_model_file:
#         model = pickle.load(trained_model_file)
#     with open(test_dataset_filename, 'rb') as test_dataset_file:
#         x_test, y_test, _ = pickle.load(test_dataset_file)
#     y_pred = model.predict(x_test)
#     return metrics.accuracy_score(y_test, y_pred)


def test_results(training_method):
    output_filename = find_output_filename(training_method)
    with open(output_filename, 'w') as output_file:
        writer_balanced = csv.writer(output_file)
        writer_balanced.writerow(["Name"] + dataset_qualities)
        for ml_model in all_models:
            trained_model_filename = find_model_filename(training_method,
                                                         ml_model)
            accuracy = dict()
            for testing_method in dataset_qualities:
                test_dataset_filename = find_dataset_filename('Test',
                                                              testing_method)
                accuracy[testing_method] = test_model(trained_model_filename,
                                                      test_dataset_filename)
            round_accuracies = [round(acc, 2)
                                for acc in [accuracy[method]
                                for method in dataset_qualities]]
            writer_balanced.writerow([ml_model + "-" + training_method] +
                                     round_accuracies)


def test_classifier(ml_model, testing_method='Augmented'):
    trained_model_filename = find_model_filename('Classification',
                                                 ml_model)
    test_dataset_filename = find_dataset_filename('Test',
                                                  testing_method)
    with open(trained_model_filename, 'rb') as trained_model_file:
        model = pickle.load(trained_model_file)
    with open(test_dataset_filename, 'rb') as test_dataset_file:
        x_test, y_test, all_timings = pickle.load(test_dataset_file)
    chosen_indices = [return_regressor_choice(model, features) for features in x_test]
    return compute_metrics(chosen_indices, y_test, all_timings)


def timings_in_test(model, testing_method='Augmented', training_method=None):
    test_dataset_filename = find_dataset_filename('Test',
                                                  testing_method)
    with open(test_dataset_filename, 'rb') as test_dataset_file:
        x_test, _, all_timings = pickle.load(test_dataset_file)
    if model == 'optimal':
        t_pred = [min(timings) for timings in all_timings]
    else:
        trained_model_filename = find_model_filename(training_method,
                                                     model)
        with open(trained_model_filename, 'rb') as trained_model_file:
            model = pickle.load(trained_model_file)
        y_pred = model.predict(x_test)
        # This doesn't work because agumenteed and balanced
        # only return one timing, not 6
        t_pred = [timings[y] for timings, y in zip(all_timings, y_pred)]
    return t_pred


def test_regressor(ml_model):
    trained_model_filename = find_model_filename('Regression',
                                                 ml_model)
    test_dataset_filename = find_dataset_filename('Test',
                                                  'Regression')
    with open(trained_model_filename, 'rb') as trained_model_file:
        model = pickle.load(trained_model_file)
    with open(test_dataset_filename, 'rb') as test_dataset_file:
        x_test, y_test, all_timings = pickle.load(test_dataset_file)
    y_pred = model.predict(x_test)
    avg_error = sum([abs(p-t) for p, t in zip(y_pred, y_test)])/len(y_pred)
    print(f"{ml_model} gave {avg_error}")


def test_model(ml_model, paradigm, testing_method='Augmented'):
    test_dataset_filename = find_dataset_filename('Test',
                                                  testing_method)
    with open(test_dataset_filename, 'rb') as test_dataset_file:
        testing_dataset = pickle.load(test_dataset_file)
    trained_model_filename = find_model_filename(paradigm,
                                                 ml_model)
    with open(trained_model_filename, 'rb') as trained_model_file:
        model = pickle.load(trained_model_file)
    chosen_indices = choose_indices(model, testing_dataset)
    return compute_metrics(chosen_indices,
                           testing_dataset['labels'],
                           testing_dataset['timings'],
                           testing_dataset['cells'],
                           ml_model)


def choose_indices(model_name, testing_dataset, paradigm='', training_quality='Augmented'):
    if model_name in heuristics:
        chosen_indices = ordering_choices_heuristics(model_name, testing_dataset, paradigm)
    else:
        trained_model_filename = find_model_filename(model_name, paradigm, training_quality)
        with open(trained_model_filename, 'rb') as trained_model_file:
            model = pickle.load(trained_model_file)
        if model_name in regressors:
            chosen_indices = [return_regressor_choice(model, features)
                              for features in testing_dataset['features']]
        elif model_name in classifiers:
            chosen_indices = [model.predict([features])[0]
                              for features in testing_dataset['features']]
        elif paradigm == 'Reinforcement':
            chosen_indices = [ordering_choice_reinforcement(model, projections)
                              for projections in testing_dataset['projections']]
    return chosen_indices


def compute_metrics(chosen_indices, testing_dataset):
    labels = testing_dataset['labels']
    all_timings = testing_dataset['timings']
    all_cells = testing_dataset['cells']
    metrics = dict()
    correct = 0
    metrics['TotalTime'] = 0
    total_markup = 0
    metrics['Completed'] = 0
    metrics['TotalCells'] = 0
    # REmove follwoign loop
    for chosen_index, label, timings, cells in \
            zip(chosen_indices, labels, all_timings, all_cells):
        if chosen_index == label:
            correct += 1
        if timings[chosen_index] not in [30, 60]:
            metrics['Completed'] += 1
        total_markup += (timings[chosen_index]-timings[label])/(timings[label] + 1)
        metrics['TotalCells'] += cells[chosen_index]
    chosen_times = [timings[index] for index, timings
                    in zip(chosen_indices, all_timings)]
    metrics['TotalTime'] = sum(chosen_times)
    total_instances = len(chosen_indices)
    metrics['Accuracy'] = correct/total_instances
    metrics['Markup'] = total_markup/total_instances
    return metrics


def return_regressor_choice(model, features):
    nvar = 3  # Make this better
    made_up_timings = list(range(math.factorial(nvar)))
    made_up_cells = list(range(math.factorial(nvar)))
    augmentated_features, _, _ = \
        augmentate_instance(features, made_up_timings, made_up_cells, nvar)
    y_op = float('inf')
    for index, x_features in enumerate(augmentated_features):
        y_pred = model.predict([x_features])
        ########
        # THIS IS NOT A LIST??
        ########
        # print(y_pred)
        if y_op > y_pred:
            y_op = y_pred
            index_op = index
    # print(index_op)
    return index_op
