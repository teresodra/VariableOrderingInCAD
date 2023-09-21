"""
The experiments in [1] are replicated with some changes.

The first change is that the testing data is balanced, so that all labels
are almost equally common.
Then we use three training sets; dataset as in [1], balanced dataset
and data augmentation dataset.

[1]Florescu, D., England, M. (2020). A Machine Learning Based Software Pipeline
to Pick the Variable Ordering for Algorithms with Polynomial Inputs.
Bigatti, A., Carette, J., Davenport, J., Joswig, M., de Wolff, T. (eds)
Mathematical Software, ICMS 2020. ICMS 2020. Lecture Notes in Computer Science,
vol 12097. Springer, Cham. https://doi.org/10.1007/978-3-030-52200-1_30
"""
import csv
from config.ml_models import ml_regressors
from create_clean_dataset import cleaning_dataset
from test_train_datasets import create_train_test_datasets
from test_train_datasets import create_regression_datasets
from choose_hyperparams import choose_hyperparams
from train_models import train_model
# from test_models import test_regressor
from test_models import test_model


# Hyperparameter tuning take a very long time,
# if tune_hyperparameters is used to decide whether to tune them
# or to used previously tuned
tune_hyperparameters = False
taking_logarithms = False

for i in range(1):
    # cleaning_dataset()
    # create_train_test_datasets()
    create_regression_datasets(taking_logarithms=taking_logarithms)

    paradigm = "regression"
    if tune_hyperparameters:
        for ml_model in ml_regressors:
            print(f"Choosing hyperparameters for {ml_model} in {paradigm}")
            choose_hyperparams(ml_model, paradigm)

    for ml_model in ml_regressors:
        print(f"Training {ml_model}")
        print(f"for {paradigm}")
        train_model(ml_model, paradigm)
    testing_method = 'augmented'
    output_file = "regression_output_acc_time.csv"
    # with open(output_file, 'a') as f:
    #     f.write("Now without logarithms and without aveg_not_zero\n")

    first_time = 1
    for ml_model in ml_regressors:
        ###
        # For KNNR running properly X.shape[0] has been changed to len(X)
        # in line 240 of
        # C:\Software\Python37\Lib\site-packages\sklearn\neighbors\_regression.py
        print(f"Testing models trained in {ml_model}")
        metrics = test_model(ml_model, paradigm=paradigm,
                             testing_method=testing_method)
        if first_time == 1:
            first_time = 0
            keys = list(metrics.keys())
            with open(output_file, 'a') as f:
                f.write('After changing dataset\n')
                f.write(', '.join(['Model'] + keys) + '\n')
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ml_model] + [metrics[key] for key in keys])
