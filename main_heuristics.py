import csv
import math
import pickle
import random
# import numpy as np
from Heuristics.heuristics_guess import not_greedy_heuristic_guess
from Heuristics.heuristics_guess import ordering_given_projections
# from find_filename import find_dataset_filename
# from test_models import compute_metrics
# from config.ml_models import heuristics

random.seed(0)

nvar = 3
testing_method = 'Biased'

# # TESTING GMODS IN AUUGMENTED : Features 2, 67 and 132
# def choose_gmods(features):
#     a = []
#     # # print(features)
#     # a.append(features[2])
#     # a.append(features[67])
#     # a.append(features[132])
#     if a[0] == min(a):
#         if a[1] <= a[2]:
#             return 0
#         else:
#             return 1
#     elif a[1] == min(a):
#         if a[0] <= a[2]:
#             return 2
#         else:
#             return 3
#     elif a[2]==min(a):
#         if a[0]<=a[1]:
#             return 4
#         else:
#             return 5


def ordering_choices_heuristics(heuristic, testing_dataset, paradigm):
    if heuristic == 'virtual-best':
        chosen_indices = testing_dataset['labels']
    elif heuristic == 'random':
        chosen_indices = [random.randint(0, len(timings)-1)
                          for timings in testing_dataset['timings']]
    else:
        if paradigm == 'Greedy':
            chosen_indices = [ordering_given_projections(projection, heuristic)
                              for projection in testing_dataset['projections']]
        elif paradigm == 'NotGreedy':
            chosen_indices = [not_greedy_heuristic_guess(polynomials,
                                                         heuristic)
                              for polynomials in testing_dataset['polynomials']]
        else:
            raise Exception(f"Paradigm {paradigm} not recognised for a heuristic.")
    return chosen_indices


# if __name__ == "__main__":
#     test_dataset_filename = find_dataset_filename('Test',
#                                                 testing_method)
#     with open(test_dataset_filename, 'rb') as test_dataset_file:
#         testing_dataset = pickle.load(test_dataset_file)
#     output_file = "heuristics_output_acc_time.csv"

#     # Testing in heuristics that make all the choice at once
#     first_heuristic = 1
#     for greedy in [True, False]:
#         for heuristic in heuristics:
#         # for heuristic in ['gmods', 'virtual best']:
#             reps = 100
#             for i in range(reps):
#                 chosen_indices = ordering_choices_heuristics(heuristic,
#                                                             testing_dataset,
#                                                             greedy=greedy)
#                 metrics = compute_metrics(chosen_indices,
#                                           testing_dataset)
#                 if i == 0:
#                     sum_metrics = metrics
#                 else:
#                     sum_metrics = {key: metrics[key] + sum_metrics[key]
#                                    for key in metrics}
#             aveg_metrics = {key: sum_metrics[key]/reps for key in sum_metrics}
#             augmented_metrics = {key: aveg_metrics[key]
#                                  if key in ['Accuracy', 'Markup']
#                                  else math.factorial(nvar)*aveg_metrics[key]
#                                  for key in sum_metrics}

#             print('not-'*(not greedy) + 'greedy-' + heuristic,
#                   augmented_metrics)
#             if first_heuristic == 1:
#                 first_heuristic = 0
#                 keys = list(augmented_metrics.keys())
#                 with open(output_file, 'a') as f:
#                     f.write(', '.join(['Model'] + keys) + '\n')
#             with open(output_file, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['not-'*(not greedy) + 'greedy-' + heuristic]
#                                 + [augmented_metrics[key] for key in keys])
