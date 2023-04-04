import yaml
from yaml import UnsafeLoader
import os
from config.ml_models import ml_models
from config.ml_models import dataset_types

print(ml_models)
for ml_model in ml_models:
    for method in dataset_types:
        filename = os.path.join(os.path.dirname(__file__),
                                'config', 'hyperparams',
                                f'{method}_{ml_model}.yaml')
        with open(filename, 'r') as f:
            hyperparameters = yaml.load(f, Loader=UnsafeLoader)
            print(type(hyperparameters), hyperparameters)
