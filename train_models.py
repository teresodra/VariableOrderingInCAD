import os
import pickle
from yaml_tools import read_yaml_from_file
from config.ml_models import classifiers


def train_model(ml_model, method):
    train_data_file = os.path.join(os.path.dirname(__file__),
                                   'datasets', 'train',
                                   f'{method}_train_dataset.txt')
    hyperparams_file = os.path.join(os.path.dirname(__file__),
                                    'config', 'hyperparams',
                                    f'{method}_{ml_model}')
    with open(train_data_file, 'rb') as f:
        method_x_train, method_y_train = pickle.load(f)
        hyperparams = read_yaml_from_file(hyperparams_file)
        current_classifier = classifiers[ml_model]
        clf = current_classifier(**hyperparams)
        clf.fit(method_x_train, method_y_train)


# print(train_model(ml_models[1], dataset_types[0]))