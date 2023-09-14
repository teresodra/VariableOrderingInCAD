'''
This file studies the metrics of the virtual_best and the random choices.
'''

from .heuristic_tools import finding_time_limit, compute_markups, compute_ncells_markup
def compute_extra_metrics(heuristic, virtual_best_timings, timings, number_no_timedout, useful_timings, ncells):

    if heuristic == 'virtual_best':
        metrics = compute_virtual_best_metrics(heuristic, virtual_best_timings)
    elif heuristic == 'random':
        metrics = compute_average_metrics(heuristic, virtual_best_timings, timings, number_no_timedout, useful_timings, ncells)
    return metrics

def compute_virtual_best_metrics(heuristic, virtual_best_timings):

    metrics = dict()
    metrics['name'] = 'virtual-best'
    no_samples = len(virtual_best_timings)
    metrics['accuracy'] = 1
    metrics['no_samples'], metrics['terminating'], metrics['timeouts_30'], metrics['timeouts_60'] = no_samples, no_samples, 0, 0
    metrics['markup'], metrics['ncells_markup'] = 0, 0
    metrics['total_time'] = sum(virtual_best_timings)
    metrics['perc_found_1'], metrics['perc_found_2'], metrics['perc_found_3'] = 1, 1, 1
    return metrics


def compute_average_metrics(heuristic, virtual_best_timings, timings, number_no_timedout, useful_timings, ncells):

    metrics = dict()
    metrics['name'] = 'random'
    no_samples = len(virtual_best_timings)
    metrics['no_samples'] = no_samples
    metrics['accuracy'] = 1/6
    prob_timeouts = [pos_timeout/6 for pos_timeout in number_no_timedout]
    metrics['terminating'] = no_samples - sum(prob_timeouts)
    metrics['timeouts_30'] = sum([prob_timeout for prob_timeout,timing in zip(prob_timeouts,timings) if finding_time_limit(timing)==30])
    metrics['timeouts_60'] = sum([prob_timeout for prob_timeout,timing in zip(prob_timeouts,timings) if finding_time_limit(timing)==60])
    expected_timings = [sum(useful_timing)/len(useful_timing) for useful_timing in useful_timings]
    metrics['markup'] = compute_markups(virtual_best_timings, expected_timings)
    metrics['ncells_markup'] = sum([sum([elem if type(elem)!=str else 10 for elem in ex_ncells])/len(ex_ncells) for ex_ncells in ncells])/len(ncells)
    metrics['total_time'] = sum(expected_timings)
    metrics['perc_found_1'], metrics['perc_found_2'], metrics['perc_found_3'] = 1/6, 2/6, 3/6

    return metrics