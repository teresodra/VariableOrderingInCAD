import pickle
from yaml_tools import read_yaml_from_file
from config.ml_models import classifiers
from find_filename import find_dataset_filename
from find_filename import find_hyperparams_filename
from find_filename import find_model_filename


def train_model(ml_model, method):
    train_data_filename = find_dataset_filename('train', method=method)
    hyperparams_file = find_hyperparams_filename(method, ml_model)
    with open(train_data_filename, 'rb') as train_data_file:
        x_train, y_train = pickle.load(train_data_file)
    hyperparams = read_yaml_from_file(hyperparams_file)
    current_classifier = classifiers[ml_model]
    clf = current_classifier(**hyperparams)
    clf.fit(x_train, y_train)
    trained_model_filename = find_model_filename(method, ml_model)
    with open(trained_model_filename, 'wb') as trained_model_file:
        pickle.dump(clf, trained_model_file)
