import math
import pickle
import random

from config.ml_models import all_models
from find_filename import find_dataset_filename
from find_filename import find_model_filename
from find_filename import find_other_filename
import numpy as np
# from sklearn import metrics
from itertools import combinations
from replicating_Dorians_features import compute_features_for_var
from test_models import compute_metrics


def train_reinforcement_model(model_name, method='Normal'):
    train_data_filename = find_dataset_filename('Train', method=method)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    # hyperparams_file = find_hyperparams_filename(method, model_name)
    # hyperparams = read_yaml_from_file(hyperparams_file)
    current_model = all_models[model_name]
    # model = current_model(**hyperparams)
    model = current_model()
    first_polys = train_dataset['projections'][0][0][0]
    first_features = get_vars_features(first_polys)
    first_labels = [random.random() for _ in range(len(first_features))]
    model.fit(first_features, first_labels)
    training_features, training_labels = [], []
    for i in range(30):
        for projections, timings \
                in zip(train_dataset['projections'], train_dataset['timings']):
            new_training_features, new_training_labels = \
                training_instances_reinforcement(model, projections, timings)
            training_features += new_training_features
            training_labels += new_training_labels
        model.fit(training_features, training_labels)
        print(test_reinforcement_model(model))
    trained_model_filename = find_model_filename('reinforcement', model_name)
    with open(trained_model_filename, 'wb') as trained_model_file:
        pickle.dump(model, trained_model_file)


def training_instances_reinforcement(model, projections, timings):
    original_polynomials = projections[0][0]
    nvar = len(original_polynomials[0][0]) - 1
    vars_features = get_vars_features(original_polynomials)
    evaluations = [model.predict([var_features])[0]
                   for var_features in vars_features]
    timing = []
    for var in range(nvar):
        # retruns the polynomials after projection wrt var
        projected_polynomials = projections[var * math.factorial(nvar-1)][1]
        new_var = var_choice_reinforcement(model, projected_polynomials)
        ordering_chosen = new_var + var * math.factorial(nvar-1)
        timing.append(timings[ordering_chosen])
    # now compute which part of the difference between
    # evaluations[i]/evaluations[j] and timing[i]/timing[j]
    # corresponds to each evaluation
    instances_features = []
    instances_labels = []
    pairs = list(combinations(range(nvar), 2))
    for i, j in pairs:
        correction_coefficient = \
            math.sqrt((timing[i]/timing[j])/(evaluations[i]/evaluations[j]))
        instances_features += [vars_features[i], vars_features[j]]
        instances_labels += [evaluations[i]*correction_coefficient,
                             evaluations[j]/correction_coefficient]
    return instances_features, instances_labels


def get_vars_features(polynomials):
    '''Will return the features of each variable
    in the given set of polynomials'''
    vars_features = []
    nvar = len(polynomials[0][0]) - 1
    unique_features_filename = find_other_filename("unique_features")
    with open(unique_features_filename, 'rb') as unique_features_file:
        unique_names = pickle.load(unique_features_file)
    for var in range(nvar):
        var_features, var_names = \
            compute_features_for_var(polynomials, var)
        var_features = [feature for feature, name
                        in zip(var_features, var_names)
                        if name in unique_names]
        vars_features.append(var_features)
    return vars_features


def var_choice_reinforcement(model, polynomials):
    '''This function will return the next variable to project
    chosen by the model trained using reinforcement'''
    vars_features = get_vars_features(polynomials)
    evaluations = model.predict(vars_features)
    min_value = np.min(evaluations)
    min_indices = np.where(evaluations == min_value)[0]
    # Randomly select one of the minimal indices
    return np.random.choice(min_indices)


def ordering_choice_reinforcement(model, projections):
    '''This function will return the ordering chosen by the RL model'''
    nvar = len(projections[0])
    ordering = 0
    for level in range(nvar-1):
        polynomials = projections[ordering][level]
        next_var = var_choice_reinforcement(model, polynomials)
        ordering += next_var * math.factorial(nvar-1-level)
    return ordering


def test_reinforcement_model(model_name, method='Normal', nvar=3):
    train_data_filename = find_dataset_filename('Test', method=method)
    with open(train_data_filename, 'rb') as train_data_file:
        testing_dataset = pickle.load(train_data_file)
    # trained_model_filename = find_model_filename('reinforcement', model_name)
    # with open(trained_model_filename, 'rb') as trained_model_file:
    #     model = pickle.load(trained_model_file)
    model = model_name
    chosen_indices = [ordering_choice_reinforcement(model, projections)
                      for projections in testing_dataset['projections']]
    metrics = compute_metrics(chosen_indices,
                              testing_dataset['labels'],
                              testing_dataset['timings'],
                              testing_dataset['cells'],
                              'reinfocement')
    augmented_metrics = {key: metrics[key] if key in ['Accuracy', 'Markup']
                         else math.factorial(nvar)*metrics[key]
                         for key in metrics}
    return augmented_metrics


train_reinforcement_model('RF-Regression')
# print(test_reinforcement_model('RFR'))
