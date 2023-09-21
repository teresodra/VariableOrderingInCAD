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
from config.ml_models import ml_models
from config.general_values import dataset_qualities
from config.general_values import purposes
from find_filename import find_dataset_filename
from find_filename import find_model_filename
from create_clean_dataset import cleaning_dataset
from test_train_datasets import create_train_test_datasets
from choose_hyperparams import choose_hyperparams
from train_models import train_model
from test_models import test_results
from test_models import timings_in_test
from test_models import test_model


# Hyperparameter tuning take a very long time,
# if tune_hyperparameters is used to decide whether to tune them
# or to used previously tuned
tune_hyperparameters = True
train_the_models = True
paradigm = 'classification'

print("MAIN.PY")
cleaning_dataset()
create_train_test_datasets()

if tune_hyperparameters:
    for ml_model in ml_models:
        for method in dataset_qualities:
            print(f"Choosing hyperparameters for {ml_model} in {method}")
            choose_hyperparams(ml_model, method)
if train_the_models:
    for ml_model in ml_models:
        print(f"Training {ml_model}")
        for method in dataset_qualities:
            print(f"for {method}")
            train_model(ml_model, method)
training_method = 'augmented'
testing_method = 'augmented'
first_time = 1
output_file = "classification_output_acc_time.csv"
for ml_model in ml_models:
    print(f"Testing models trained in {training_method}")
    metrics = test_model(ml_model,
                         paradigm=training_method,
                         testing_method=testing_method)
    if first_time == 1:
        first_time = 0
        keys = list(metrics.keys())
        with open(output_file, 'a') as f:
            f.write('No hyperparameters\n')
            f.write(', '.join(['Model'] + keys) + '\n')
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ml_model] + [metrics[key] for key in keys])


# timings = dict()
# testing_method = 'augmented'
# test_dataset_filename = find_dataset_filename('Test',
#                                               testing_method)

# with open("classification_output_timings.csv", 'w') as f:
#     f.write("model, Normal, Balanced, Augmented\n")
# for ml_model in ml_models:
#     for training_method in dataset_qualities:
#         trained_model_filename = find_model_filename(training_method,
#                                                      ml_model)
#         accuracy = test_model(trained_model_filename,
#                               test_dataset_filename)
#         timings[training_method] = timings_in_test(ml_model, testing_method,
#                                                    training_method)
#         total_time = sum(timings[training_method])
#         # with open("classification_output_acc_time.csv", 'a') as f:
#         #     f.write(f"{ml_model}, {accuracy}, {total_time}\n")
#     with open("classification_output_timings.csv", 'a') as f:
#         f.write(f"{ml_model}, {sum(timings['Normal'])}, \
#                  {sum(timings['Balanced'])}, {sum(timings['Augmented'])}\n")
#     timings['optimal'] = timings_in_test('optimal', testing_method)
#     print(sum(timings['optimal']))
#     from make_plots import survival_plot
#     survival_plot(timings, plot_name=f"survival_plot_{ml_model}")
