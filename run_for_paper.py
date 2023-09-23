import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from create_clean_dataset import cleaning_dataset
from test_train_datasets import create_train_test_datasets
from test_train_datasets import create_regression_datasets
from config.ml_models import all_models
from config.ml_models import regressors
from config.ml_models import classifiers
from config.ml_models import heuristics
from choose_hyperparams import choose_hyperparams
from train_models import train_model
from main_heuristics import ordering_choices_heuristics
from find_filename import find_dataset_filename
# from find_filename import find_timings_lists
from find_filename import find_hyperparams_filename
from find_filename import find_all_info
from test_models import compute_metrics
from test_models import choose_indices


# def metrics_for_all_reps(all_indices_chosen, testing_dataset, ml_model):
#     all_metrics = [compute_metrics(chosen_indices, testing_dataset)
#                    for chosen_indices in all_indices_chosen]
#     aveg_metrics = {key: sum(metrics[key]/len(all_metrics)
#                              for metrics in all_metrics)
#                     for key in all_metrics[0]}
#     all_timings = testing_dataset['timings']
#     aveg_timings = []
#     for instance in range(len(all_indices_chosen[0])):
#         instance_timings = [timings[indices_chosen[instance]]
#                             for timings, indices_chosen
#                             in zip(all_timings, all_indices_chosen)]
#         aveg_timings.append(instance_timings)
#     timings_lists_filename = find_timings_lists(ml_model)
#     with open(timings_lists_filename, 'wb') as timings_lists_file:
#         pickle.dump(aveg_timings, timings_lists_file)
#     all_total_times = [metrics['TotalTime'] for metrics in all_metrics]
#     return aveg_metrics, all_total_times


def dominiks_plots(all_total_times):
    data = []
    for key in all_total_times:
        data.extend([{'Model': key, 'Total time': total_time}
                     for total_time in all_total_times[key]])
    df = pd.DataFrame(data)

    # Create a box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Model', y='Total time', data=df)

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Total time')
    plt.title('Model Total time Comparison')

    # Display the plot
    plt.show()


def repeat_instances_dataset(dataset, n_reps):
    new_dataset = dict()
    for key in dataset:
        new_dataset[key] = [elem for elem in dataset[key]
                            for _ in range(n_reps)]
    return new_dataset


def study_a_model(model_name: str,
                  testing_quality: str,
                  paradigm: str,
                  training_quality: str = '',
                  tune_hyperparameters: bool = False,
                  reps: int = 10
                  ):
    if model_name in heuristics:
        if training_quality != '':
            raise Exception(f"training_quality cannot be {training_quality}.")
        if tune_hyperparameters is not False:
            raise Exception(f"Hyperparams cannot be tuned for {paradigm}.")
    testing_filename = find_dataset_filename('Test', testing_quality)
    with open(testing_filename, 'rb') as testing_file:
        testing_dataset = pickle.load(testing_file)
    if testing_quality == 'Biased':
        # If the dataset contains less factorial_nvar less instances,
        # we repeat each instance factorial_nvar times
        factorial_nvar = len(testing_dataset['projections'][0])
        testing_dataset = \
            repeat_instances_dataset(testing_dataset, factorial_nvar)
    all_metrics = []
    all_timings = []
    for _ in range(reps):
        if model_name not in heuristics:
            # If the paradigm is 'Heuristics' there is no need
            # to tune_hyperparameters or to train the models
            hyperparams_filename = find_hyperparams_filename(model_name,
                                                             paradigm,
                                                             training_quality) + '.yaml'
            if tune_hyperparameters or not os.path.exists(hyperparams_filename):
                if not os.path.exists(hyperparams_filename):
                    print('hyperparams_filename doesnt exits \n', hyperparams_filename)
                choose_hyperparams(model_name, paradigm, training_quality)
            # Hyperparameters ready
            train_model(model_name, paradigm, training_quality)
            # Model trained
        chosen_indices = choose_indices(model_name, testing_dataset,
                                        paradigm, training_quality)
        # Indices chosen by the model
        all_metrics.append(compute_metrics(chosen_indices, testing_dataset))
        all_timings.append([timings[index] for timings, index
                            in zip(testing_dataset['timings'],
                                   chosen_indices)])
    model_info = dict()
    model_info['AverageMetrics'] = {key: sum(metrics[key] for metrics
                                             in all_metrics)/reps
                                    for key in all_metrics[0]}
    # average metrics computed for comparison purposes
    model_info['AverageTimings'] = [sum(all_timings_in_instance)/reps
                                    for all_timings_in_instance
                                    in zip(*all_timings)]
    # average timings in each instance to create adversarial plots
    for key in all_metrics[0]:
        model_info['All' + key] = [metrics[key]
                                   for metrics in all_metrics]
    # info of all metrics saved for seaborn boxplots
    all_info_filename = find_all_info(model_name, paradigm, training_quality)
    with open(all_info_filename, 'wb') as all_info_file:
        pickle.dump(model_info, all_info_file)
    return model_info


if __name__ == "__main__":
    reps = 1
    data = dict()
    data['TotalTime'] = []
    new_datasets = True
    if new_datasets:
        cleaning_dataset()
        create_train_test_datasets()
        create_regression_datasets()
    all_total_times = dict()
    for model_name in list(all_models) + heuristics:
        if model_name in heuristics:
            testing_quality = 'Biased'
            training_quality = ''
            tune_hyperparameters = False
            paradigm = 'Greedy'  # NotGreedy
        else:
            testing_quality = 'Augmented'
            training_quality = 'Augmented'
            tune_hyperparameters = False
            if model_name in classifiers:
                paradigm = ''
            elif model_name in regressors:
                paradigm = 'Regression'
        print(model_name)
        model_info = study_a_model(model_name=model_name,
                                   testing_quality=testing_quality,
                                   paradigm=paradigm,
                                   training_quality=training_quality,
                                   tune_hyperparameters=tune_hyperparameters,
                                   reps=reps
                                   )
        all_total_times[model_name] = model_info['AllTotalTime']

    dominiks_plots(all_total_times)




# def choose_indices(model, dataset):
#     if model in classifiers:
#     elif model in heuristics:
#         ordering_choices_heuristics(model, dataset)
