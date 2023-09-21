import math
import pickle
import random
from yaml_tools import read_yaml_from_file
from config.ml_models import all_models
from find_filename import find_dataset_filename
from find_filename import find_hyperparams_filename
from find_filename import find_model_filename
from find_filename import find_other_filename
from dataset_manipulation import give_all_symmetries
import numpy as np
# from sklearn import metrics
from itertools import combinations
from replicating_Dorians_features import compute_features_for_var
from test_models import compute_metrics


def train_model(ml_model, method):
    train_data_filename = find_dataset_filename('Train', method=method)
    hyperparams_file = find_hyperparams_filename(method, ml_model)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    hyperparams = read_yaml_from_file(hyperparams_file)
    current_model = all_models[ml_model]
    model = current_model(**hyperparams)
    # model = current_model()
    print('here')
    model.fit(train_dataset['features'], train_dataset['labels'])
    trained_model_filename = find_model_filename(method, ml_model)
    print('here2')
    with open(trained_model_filename, 'wb') as trained_model_file:
        pickle.dump(model, trained_model_file)
    return model


def train_regression_model(ml_model, method):
    train_data_filename = find_dataset_filename('Train', method=method)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    # hyperparams_file = find_hyperparams_filename(method, ml_model)
    # hyperparams = read_yaml_from_file(hyperparams_file)
    train_dataset['features'] = np.asarray([x_t for x_t, t_t in zip(train_dataset['features'], train_dataset['timings'])
                                            if t_t[:4] != 'Over'], dtype=float)
    train_dataset['timings'] = np.asarray([t_t for t_t in train_dataset['timings']
                                           if t_t[:4] != 'Over'], dtype=float)
                          ####
                          # IS THIS REALLY DOING SOMTHING?
                          # What if we used twice timelimit instead
    current_model = ml_regressors[ml_model]
    reg = current_model()  # **hyperparams)
    reg.fit(train_dataset['features'], train_dataset['timings'])
    # trained_model_filename = find_model_filename(method, ml_model, 'regression')
    # with open(trained_model_filename, 'wb') as trained_model_file:
    #     pickle.dump(reg, trained_model_file)
    return reg


def choose_using_regression(x_test, regressor):
    timings = regressor.predict(give_all_symmetries(x_test, 0))
    return np.argmin(timings)


def test_regression_model(method, regressor):
    test_data_filename = find_dataset_filename('Test', method=method)
    with open(test_data_filename, 'rb') as test_data_file:
        x_test, y_test, t_test = pickle.load(test_data_file)
    x_test = np.asarray([x_t for x_t, t_t in zip(x_test, t_test)
                         if t_t[:4] != 'Over'], dtype=float)
    y_test = np.asarray([y_t for y_t, t_t in zip(y_test, t_test)
                         if t_t[:4] != 'Over'], dtype=float)
    y_pred = [choose_using_regression(x_i, regressor) for x_i in x_test]


def train_reinforcement_model(ml_model, method='Normal'):
    train_data_filename = find_dataset_filename('Train', method=method)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    # hyperparams_file = find_hyperparams_filename(method, ml_model)
    # hyperparams = read_yaml_from_file(hyperparams_file)
    current_model = all_models[ml_model]
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
    trained_model_filename = find_model_filename('reinforcement', ml_model)
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


def test_reinforcement_model(ml_model, method='Normal', nvar=3):
    train_data_filename = find_dataset_filename('Test', method=method)
    with open(train_data_filename, 'rb') as train_data_file:
        testing_dataset = pickle.load(train_data_file)
    # trained_model_filename = find_model_filename('reinforcement', ml_model)
    # with open(trained_model_filename, 'rb') as trained_model_file:
    #     model = pickle.load(trained_model_file)
    model = ml_model
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
