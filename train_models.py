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
    train_data_filename = find_dataset_filename('train', method=method)
    hyperparams_file = find_hyperparams_filename(method, ml_model)
    with open(train_data_filename, 'rb') as train_data_file:
        if method == "Normal":
            x_train, y_train, _, _ = pickle.load(train_data_file)
        else:
            x_train, y_train, _ = pickle.load(train_data_file)
            # a = pickle.load(train_data_file)
            # print(a[0], type(a), len(a), method)
    hyperparams = read_yaml_from_file(hyperparams_file)
    current_classifier = sklearn_models[ml_model]
    clf = current_classifier(**hyperparams)
    clf.fit(x_train, y_train)
    trained_model_filename = find_model_filename(method, ml_model)
    with open(trained_model_filename, 'wb') as trained_model_file:
        pickle.dump(clf, trained_model_file)


def train_regression_model(ml_model, method):
    train_data_filename = find_dataset_filename('train', method=method)
    with open(train_data_filename, 'rb') as train_data_file:
        x_train, _, t_train = pickle.load(train_data_file)
    # hyperparams_file = find_hyperparams_filename(method, ml_model)
    # hyperparams = read_yaml_from_file(hyperparams_file)
    x_train = np.asarray([x_t for x_t, t_t in zip(x_train, t_train)
                          if t_t[:4] != 'Over'], dtype=float)
    t_train = np.asarray([t_t for t_t in t_train 
                          if t_t[:4] != 'Over'], dtype=float)
    current_classifier = regressors[ml_model]
    # print(t_train)
    print("her")
    reg = current_classifier()  # **hyperparams)
    reg.fit(x_train, t_train)
    # trained_model_filename = find_model_filename(method, ml_model, 'regression')
    # with open(trained_model_filename, 'wb') as trained_model_file:
    #     pickle.dump(reg, trained_model_file)
    print("Real")
    print(t_train[10:20])
    print("Predicted")
    print(reg.predict(x_train)[10:20])
    print(metrics.mean_squared_error(reg.predict(x_train), t_train))
    return reg


def choose_using_regression(x_test, regressor):
    timings = regressor.predict(give_all_symmetries(x_test, 0))
    return np.argmin(timings)


def test_regression_model(method, regressor):
    test_data_filename = find_dataset_filename('test', method=method)
    with open(test_data_filename, 'rb') as test_data_file:
        x_test, y_test, t_test = pickle.load(test_data_file)
    x_test = np.asarray([x_t for x_t, t_t in zip(x_test, t_test)
                         if t_t[:4] != 'Over'], dtype=float)
    y_test = np.asarray([y_t for y_t, t_t in zip(y_test, t_test)
                         if t_t[:4] != 'Over'], dtype=float)
    y_pred = [choose_using_regression(x_i, regressor) for x_i in x_test]
    print("ACC", metrics.accuracy_score(y_test, y_pred))


# for ml_reg in ml_regressors:
#     print(ml_reg)
#     regressor = train_regression_model(ml_reg, 'balanced')
#     print(ml_reg)
#     test_regression_model('balanced', regressor)
