import csv
import math
import pickle
import random
# import numpy as np
from Heuristics.heuristics_guess import not_greedy_heuristic_guess
from Heuristics.heuristics_guess import choose_order_given_projections
from find_filename import find_dataset_filename
from test_models import compute_metrics

random.seed(0)

nvar = 3
testing_method = 'Normal'
test_dataset_filename = find_dataset_filename('Test',
                                              testing_method)
with open(test_dataset_filename, 'rb') as test_dataset_file:
    testing_dataset = pickle.load(test_dataset_file)
output_file = "heuristics_output_acc_time.csv"


# TESTING GMODS IN AUUGMENTED : Features 2, 67 and 132
def choose_gmods(features):
    a = []
    # # print(features)
    # a.append(features[2])
    # a.append(features[67])
    # a.append(features[132])
    if a[0]==min(a):
        if a[1]<=a[2]:
            return 0
        else:
            return 1
    elif a[1]==min(a):
        if a[0]<=a[2]:
            return 2
        else:
            return 3
    elif a[2]==min(a):
        if a[0]<=a[1]:
            return 4
        else:
            return 5


# Testing in heuristics that make all the choice at once
first_heuristic = 1
for heuristic in ['T1', 'gmods', 'brown', 'random', 'virtual-best']:
# for heuristic in ['gmods', 'virtual best']:
    reps = 100
    sum_metrics = dict()
    for i in range(reps):
        if heuristic == 'virtual-best':
            # chosen_indices = [np.argmin(timings) for timings in testing_dataset['timings']]
            chosen_indices = testing_dataset['labels']
        elif heuristic == 'random':
            chosen_indices = [random.randint(0, 5) for timings in testing_dataset['timings']]
        else:
            chosen_indices = [not_greedy_heuristic_guess(projection[0][0], heuristic)
                              for projection in testing_dataset['projections']]
            # chosen_indices = [choose_gmods(features)
            #                   for features in testing_dataset['features']]
        metrics = compute_metrics(chosen_indices,
                                  testing_dataset['labels'],
                                  testing_dataset['timings'],
                                  testing_dataset['cells'],
                                  heuristic)
        if len(sum_metrics) == 0:
            sum_metrics = metrics
        else:
            sum_metrics = {key: metrics[key] + sum_metrics[key] for key in metrics}
    aveg_metrics = {key: sum_metrics[key]/reps for key in sum_metrics}
    augmented_metrics = {key: aveg_metrics[key] if key in ['Accuracy', 'Markup'] else math.factorial(nvar)*aveg_metrics[key] for key in sum_metrics}

    print(heuristic, augmented_metrics)
    if first_heuristic == 1:
        first_heuristic = 0
        keys = list(augmented_metrics.keys())
        with open(output_file, 'a') as f:
            f.write('Choosing the whole ordering in the beggining \n')
            f.write(', '.join(['Model'] + keys) + '\n')
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([heuristic] + [augmented_metrics[key] for key in keys])

# # Testing on greedy heuristics
# for heuristic in ['brown', 'gmods', 'random', 'virtual best']:
#     reps = 100
#     sum_metrics = dict()
#     for i in range(reps):
#         if heuristic == 'virtual best':
#             chosen_indices = [np.argmin(timings) for timings in testing_dataset['timings']]
#         elif heuristic == 'random':
#             chosen_indices = [random.randint(0, 5) for timings in testing_dataset['timings']]
#         else:
#             chosen_indices = [choose_order_given_projections(projection, heuristic)
#                               for projection in testing_dataset['projections']]
#         metrics = compute_metrics(chosen_indices,
#                                   testing_dataset['labels'],
#                                   testing_dataset['timings'],
#                                   testing_dataset['cells'],
#                                   heuristic)
#         if len(sum_metrics) == 0:
#             sum_metrics = metrics
#         else:
#             sum_metrics = {key: metrics[key] + sum_metrics[key] for key in metrics}
#     aveg_metrics = {key: sum_metrics[key]/reps for key in sum_metrics}
#     augmented_metrics = {key: aveg_metrics[key] if key in ['Accuracy', 'Markup'] else math.factorial(nvar)*aveg_metrics[key] for key in sum_metrics}
        
#     print(heuristic, augmented_metrics)
#     if first_heuristic == 1:
#         first_heuristic = 0
#         keys = list(augmented_metrics.keys())
#         with open(output_file, 'a') as f:
#             f.write('Now choosing greedily \n')
#             f.write(', '.join(['Model'] + keys) + '\n')
#     with open(output_file, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([heuristic] + [augmented_metrics[key] for key in keys])
# # print(sum(min(timings) for timings in testing_dataset['timings']))
