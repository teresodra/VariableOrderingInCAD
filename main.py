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
from test_models import timings_in_test


# Hyperparameter tuning take a very long time,
# if tune_hyperparameters is used to decide whether to tune them
# or to used previously tuned
# tune_hyperparameters = False


# cleaning_dataset()
# create_train_test_datasets()

# if tune_hyperparameters:
#     for ml_model in ml_models:
#         for method in dataset_types:
#             print(f"Choosing hyperparameters for {ml_model} in {method}")
#             choose_hyperparams(ml_model, method)
# for ml_model in ml_models:
#     print(f"Training {ml_model}")
#     for method in dataset_types:
#         print(f"for {method}")
#         train_model(ml_model, method)
# for training_method in dataset_types:
#     print(f"Testing models trained in {training_method}")
#     test_results(training_method)

timings = dict()
model = 'SVC'
testing_method = 'Augmented'
for training_method in dataset_types:
    print(f"Testing models trained in {training_method}")
    timings[training_method] = timings_in_test(model, testing_method, training_method)

from make_plots import survival_plot

survival_plot(timings)