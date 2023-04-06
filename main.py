"""
The experiments in [1] are replicated with some changes.

The first change is that the testing data is balanced, so that all targets
are almost equally common.
Then we use three training sets; dataset as in [1], balanced dataset
and data augmentation dataset.

[1]Florescu, D., England, M. (2020). A Machine Learning Based Software Pipeline
to Pick the Variable Ordering for Algorithms with Polynomial Inputs.
Bigatti, A., Carette, J., Davenport, J., Joswig, M., de Wolff, T. (eds)
Mathematical Software, ICMS 2020. ICMS 2020. Lecture Notes in Computer Science,
vol 12097. Springer, Cham. https://doi.org/10.1007/978-3-030-52200-1_30
"""
from config.ml_models import ml_models
from config.ml_models import dataset_types
from find_filename import find_dataset_filename
from create_clean_dataset import cleaning_dataset
from test_train_datasets import create_train_test_datasets
from choose_hyperparams import choose_hyperparams
from train_models import train_model
from test_models import test_results


original_dataset_file = find_dataset_filename('unclean')
clean_dataset_filename = find_dataset_filename('clean')
cleaning_dataset(original_dataset_file, clean_dataset_filename)
create_train_test_datasets()

for ml_model in ml_models:
    for method in dataset_types:
        choose_hyperparams(ml_model, method)
for ml_model in ml_models:
    for method in dataset_types:
        train_model(ml_model, method)
for testing_method in ['normal', 'balanced']:
    test_results(testing_method)
