import pickle
from yaml_tools import read_yaml_from_file
from config.ml_models import sklearn_models
from config.ml_models import ml_regressors
from find_filename import find_dataset_filename
from find_filename import find_hyperparams_filename
from find_filename import find_model_filename
from dataset_manipulation import give_all_symmetries
import numpy as np
from sklearn import metrics


def train_model(ml_model, method):
    train_data_filename = find_dataset_filename('Train', method=method)
    hyperparams_file = find_hyperparams_filename(method, ml_model)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    hyperparams = read_yaml_from_file(hyperparams_file)
    current_classifier = sklearn_models[ml_model]
    clf = current_classifier(**hyperparams)
    print("DATaset", train_dataset.keys())
    clf.fit(train_dataset['features'], train_dataset['labels'])
    trained_model_filename = find_model_filename(method, ml_model)
    with open(trained_model_filename, 'wb') as trained_model_file:
        pickle.dump(clf, trained_model_file)


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
    current_classifier = ml_regressors[ml_model]
    # print(train_dataset['timings'])
    print("her")
    reg = current_classifier()  # **hyperparams)
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
