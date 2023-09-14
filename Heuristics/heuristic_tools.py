'''
This file contains a variety of functions useful for other functions.
'''

import itertools
import os
import pickle
import math
import json

from numpy import Inf

all_heuristics = ['sumsignsumdeg','sotd', 'old_mods', 'mods', 'logmods', 'acc_logmods', 'greedy_sotd', 'brown', 'gmods', 'greedy_logmods', 'random', 'virtual_best']
for degree_type in ['svdeg', 'signdeg', 'deg']:
    for monomial_operation in ['sum','max','aveg']:
        for polynomial_operation in ['sum','max','aveg']:
            all_heuristics += [polynomial_operation + monomial_operation + degree_type]
    
expensive_heuristics = ['sotd', 'old_mods', 'mods', 'logmods', 'acc_logmods']
greedy_heuristics = [heuristic for heuristic in all_heuristics if heuristic not in expensive_heuristics]
not_heuristics = ['virtual_best', 'random']



def create_pseudorderings(nvar):
    '''
    This function returns the possible orderings for CAD but written in a different fashion
    that makes more sense for the way projections are saved
    '''
    l = [list(range(nvar-i)) for i in range(nvar)]
    return itertools.product(*l)


def compute_real_timings(timings, choice_timings_including_str, virtual_best_timings, max_penalization_if_not_finished=Inf):
    '''
    This function returns the timing that are used for computing the metrics
    '''
    return [ choice_time if type(choice_time) is not str else min(max_penalization_if_not_finished*virtual_best_time,2*finding_time_limit(timing)) for timing, choice_time, virtual_best_time in zip(timings, choice_timings_including_str, virtual_best_timings)]

def compute_markups(virtual_best_timings, real_timings, smoother = 1):
    ''' 
    This function computes the markups of the chosen orderings
    with respect to the virtual_best orderings.
    The smoother is a parameter applied to avoid unreasonable markups.
    '''
    # the especified smoother is used
    
    markups = [(choice_timing-virtual_best_timing)/(virtual_best_timing+smoother) for virtual_best_timing, choice_timing in zip(virtual_best_timings, real_timings)] 
    return sum(markups)/len(markups)
    
def compute_ncells_markup(ncells, guesses,ncell_markup_default = 10):
    ncells_markups = [ncell[guess]/min([nc for nc in ncell if type(nc)!=str])-1 if type(ncell[guess])!=str else ncell_markup_default for ncell, guess in zip(ncells, guesses)]
    ncells_markup = sum(ncells_markups)/len(ncells_markups)
    return ncells_markup

def order_mate(order):
    '''Returns the order that shares the first variable projected with the given order'''
    if order == 0:
        return 1
    elif order == 1:
        return 0
    elif order == 2:
        return 3
    elif order == 3:
        return 2
    elif order == 4:
        return 5
    elif order == 5:
        return 4
    else:
        raise Exception('Order too big')


def finding_time_limit(timings):
    '''
    Returns the timelimit that was given.
    '''
    if min([timing for timing in timings if type(timing) is not str])>30:
        return 60
    else:
        return 30


def minimum_indices(given_list):
    '''
    Returns the indices containing the minima of a list.
    Helpful function for the function above
    '''
    minimum = min(given_list)
    return [index for index, value in enumerate(given_list) if value == minimum]


def multiplyList(myList) :
    '''
    Multiplies all the elements in a list
    '''
    result = 1
    for x in myList:
         result = result * x
    return result


def all_combinations(l):
    '''
    Returns all possible combinations of a given list.
    More concretely, all possible subsets ordered in all possible ways.
    '''
    combs_with_order = []
    for i in range(1,len(l)+1):
        combs=list(itertools.combinations(l,i))
        for comb in combs:
            combs_with_order+=list(itertools.permutations(comb,i))
    return combs_with_order


def all_combinations_fixed_length(l, i):
    '''
    Returns all possible combinations of a given list.
    More concretely, all possible subsets ordered in all possible ways.
    '''
    combs_with_order = []
    combs=list(itertools.combinations(l,i))
    for comb in combs:
        combs_with_order+=list(itertools.permutations(comb,i))
    return combs_with_order


def trim_dataset(dataset, minimum_time_to_consider=0):
    '''
    Returns the dataset containing only the problems that took 
    at least 'minimum_time_to_consider' seconds to finish.
    '''
    projections, targets, timings, heuristics_costs, ncells = dataset
    new_projections = [projection for projection, timing, target in zip(projections, timings, targets) if timing[target]>minimum_time_to_consider]
    new_targets = [target for target, timing in zip(targets, timings) if timing[target]>minimum_time_to_consider]
    new_timings = [timing for timing, target in zip(timings, targets) if timing[target]>minimum_time_to_consider]
    new_heuristics_costs = [heuristics_cost for heuristics_cost, timing, target in zip(heuristics_costs, timings, targets) if timing[target]>minimum_time_to_consider]
    new_ncells = [ncells for ncells, timing, target in zip(ncells, timings, targets) if timing[target]>minimum_time_to_consider]

    return new_projections, new_targets, new_timings, new_heuristics_costs, new_ncells


def get_dataset(without_repetition=True, return_ncells=True, minimum_time_to_consider=0):
    '''
    Uploads the desired dataset from its location
    '''

    if without_repetition:
        aux_name = 'without_repetition'
    else:
        aux_name = 'with_repetition'

    if return_ncells:
        dataset_location = os.path.join(os.path.dirname(__file__), '..','Datasets','ThreeVariableSMTLIB2021','dataset_'+aux_name+'_return_ncells.txt')
        #dataset_location = 'C:\\Users\\delriot\\OneDrive - Coventry University\\03Repositories\\01DEWCADCoventry\\Datasets\\dataset_without_repetition_return_ncells.txt'
    else:
        dataset_location = os.path.join(os.path.dirname(__file__), '..','Datasets','dataset_'+aux_name+'.txt')

    f = open(dataset_location, 'rb')
    dataset = pickle.load(f)
    f.close()

    return trim_dataset(dataset, minimum_time_to_consider=minimum_time_to_consider)

   
def aveg_of_not_zero(given_list):
    '''
    Takes the average of a list without considering the elements that are 0.'''
    s= sum(given_list)
    if s>0:
       return s/sum([1 for elem in given_list if elem>0])
    else:
        return 0

def substract_two_timings(time1, time2):
    '''time1 minus time2'''
    if type(time1) is str and type(time2) is str:
        return 0
    elif type(time1) is str and type(time2) is not str:
        return 30
    elif type(time1) is not str and type(time2) is str:
        return -30
    elif type(time1) is not str and type(time2) is not str:
        return time1-time2

# This is how to save the best features
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_features')
# best_features = ['summaxdeg', 'avegavegdeg', 'sumsumdeg', 'avegavegsigndeg' , 'sumsignsumdeg', 'summaxsvdeg']# , 'sumsumsigndeg', 'sumsumsvdeg'
# with open(file_path, 'w') as file:
#     json.dump(best_features, file)

# This is how to load the best features
with open(file_path, 'r') as file:
    best_features = json.load(file)

paper_all_pos = all_combinations(best_features)
indices = list(range(len(best_features)))
paper_all_indices = [str(elem).replace(', ','>').replace('(','').replace(')','') for elem in all_combinations(indices)]
existing_heuristics = ['brown', 'mods', 36, 'random', 'virtual_best'] # 36 is gmods
survival_plot_heuristics = ['virtual_best', 36, 'brown']
ml_models = []

