import math
import pickle
from yaml_tools import read_yaml_from_file
from config.ml_models import sklearn_models
from config.ml_models import ml_regressors
from find_filename import find_dataset_filename
from find_filename import find_hyperparams_filename
from find_filename import find_model_filename
from find_filename import find_other_filename
from dataset_manipulation import give_all_symmetries
import numpy as np
from sklearn import metrics
from itertools import combinations
from replicating_Dorians_features import compute_features_for_var


def train_model(ml_model, method):
    train_data_filename = find_dataset_filename('Train', method=method)
    hyperparams_file = find_hyperparams_filename(method, ml_model)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    hyperparams = read_yaml_from_file(hyperparams_file)
    current_model = sklearn_models[ml_model]
    model = current_model(**hyperparams)
    # model = current_model()
    model.fit(train_dataset['features'], train_dataset['labels'])
    trained_model_filename = find_model_filename(method, ml_model)
    with open(trained_model_filename, 'wb') as trained_model_file:
        pickle.dump(model, trained_model_file)


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
    print("Real")
    print(train_dataset['timings'][10:20])
    print("Predicted")
    print(reg.predict(train_dataset['features'])[10:20])
    print(metrics.mean_squared_error(reg.predict(train_dataset['features']), train_dataset['timings']))
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


# for ml_reg in ml_regressors:
#     print(ml_reg)
#     regressor = train_regression_model(ml_reg, 'balanced')
#     print(ml_reg)
#     test_regression_model('balanced', regressor)

def train_reinforcement_model(ml_model, method='Augmented'):
    train_data_filename = find_dataset_filename('Train', method=method)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    hyperparams_file = find_hyperparams_filename(method, ml_model)
    hyperparams = read_yaml_from_file(hyperparams_file)
    current_model = sklearn_models[ml_model]
    model = current_model(**hyperparams)
    for projections, timings \
            in zip(train_dataset['projections'], train_dataset['timings']):
        training_features, training_labels = \
            training_instances_reinforcement(model, projections)
        model.fit(training_features, training_labels)
    


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
        timing[var] = timings[ordering_chosen]
    # now compute which part of the difference between 
    # evaluations[i]/evaluations[j] and timing[i]/timing[j]
    # corresponds to each evaluation
    instances_features = []
    instances_labels = []
    pairs = list(combinations(range(nvar), 2))
    for i, j in pairs:
        correction_coefficient = \
            math.sqrt((timing[j]/timing[j])/(evaluations[i]/evaluations[j]))
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
    with open(unique_features_filename, 'wb') as unique_features_file:
        unique_names = pickle.load(unique_features_file)
    print(unique_names)
    for var in range(nvar):
        var_features, var_names = \
            compute_features_for_var(polynomials, var)
        var_features = [feature for feature, name
                        in zip(var_features, var_names)
                        if name in unique_names]
        vars_features += var_features
    return vars_features


def var_choice_reinforcement(model, polynomials):
    '''This function will return the next variable to project chosen by the model trained using reinforcement'''
    vars_features = get_vars_features(model, polynomials)
    evaluations = [model.predict([var_features])[0]
                   for var_features in vars_features]
    return evaluations.index(min(evaluations))
