
import pickle
from yaml_tools import read_yaml_from_file
from config.ml_models import all_models
from find_filename import find_dataset_filename
from find_filename import find_hyperparams_filename
from find_filename import find_model_filename
# from find_filename import find_other_filename
# from dataset_manipulation import give_all_symmetries
# import numpy as np
# from sklearn import metrics
# from itertools import combinations
# from replicating_Dorians_features import compute_features_for_var
# from test_models import compute_metrics


def train_model(model_name, paradigm, training_quality):
    print(f"Training {model_name}")
    train_data_filename = find_dataset_filename('Train', dataset_quality=training_quality, paradigm=paradigm)
    hyperparams_file = find_hyperparams_filename(model_name, paradigm=paradigm, training_quality=training_quality)
    with open(train_data_filename, 'rb') as train_data_file:
        train_dataset = pickle.load(train_data_file)
    hyperparams = read_yaml_from_file(hyperparams_file)
    current_model = all_models[model_name]
    model = current_model(**hyperparams)
    # model = current_model()
    model.fit(train_dataset['features'], train_dataset['labels'])
    trained_model_filename = find_model_filename(model_name, paradigm, training_quality)
    with open(trained_model_filename, 'wb') as trained_model_file:
        pickle.dump(model, trained_model_file)
    return model


# def train_regression_model(model_name, method):
#     train_data_filename = find_dataset_filename('Train', method=method)
#     with open(train_data_filename, 'rb') as train_data_file:
#         train_dataset = pickle.load(train_data_file)
#     # hyperparams_file = find_hyperparams_filename(method, model_name)
#     # hyperparams = read_yaml_from_file(hyperparams_file)
#     train_dataset['features'] = np.asarray([x_t for x_t, t_t in zip(train_dataset['features'], train_dataset['timings'])
#                                             if t_t[:4] != 'Over'], dtype=float)
#     train_dataset['timings'] = np.asarray([t_t for t_t in train_dataset['timings']
#                                            if t_t[:4] != 'Over'], dtype=float)
#                           ####
#                           # IS THIS REALLY DOING SOMTHING?
#                           # What if we used twice timelimit instead
#     current_model = ml_regressors[model_name]
#     reg = current_model()  # **hyperparams)
#     reg.fit(train_dataset['features'], train_dataset['timings'])
#     # trained_model_filename = find_model_filename(method, model_name, 'regression')
#     # with open(trained_model_filename, 'wb') as trained_model_file:
#     #     pickle.dump(reg, trained_model_file)
#     return reg


# def choose_using_regression(x_test, regressor):
#     timings = regressor.predict(give_all_symmetries(x_test, 0))
#     return np.argmin(timings)


# def test_regression_model(method, regressor):
#     test_data_filename = find_dataset_filename('Test', method=method)
#     with open(test_data_filename, 'rb') as test_data_file:
#         x_test, y_test, t_test = pickle.load(test_data_file)
#     x_test = np.asarray([x_t for x_t, t_t in zip(x_test, t_test)
#                          if t_t[:4] != 'Over'], dtype=float)
#     y_test = np.asarray([y_t for y_t, t_t in zip(y_test, t_test)
#                          if t_t[:4] != 'Over'], dtype=float)
#     y_pred = [choose_using_regression(x_i, regressor) for x_i in x_test]
