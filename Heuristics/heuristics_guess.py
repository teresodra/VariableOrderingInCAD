'''
This file contains the functions that given all projections with 
all possible orderings, return the ordering that would have been
choose by the desired heuristic.
'''

import random
from math import factorial
from .heuristics_rules import *
from .heuristic_tools import greedy_heuristics, expensive_heuristics, create_pseudorderings, ml_models


def ordering_given_projections(projections, method="gmods"):
    '''Returns the order guessed by the heuristic requested'''
    if method in greedy_heuristics or type(method) == int or method == 'T1':
        guess = greedy_heuristic_guess(projections, heuristic=method)
        return guess
    elif method in expensive_heuristics:
        return no_greedy_heuristic_guess(projections, heuristic=method)
    elif method in ml_models:
        return ml_model_guess(projections, method=method)
    else:
        raise Exception(f'Heuristic not recognised:{method}.')


def greedy_heuristic_guess(projections:list, heuristic:str="gmods"):
    '''
    This function is especialized in greedy heuristics.
    One variable is picked at a time, adjusting the ordering accordingly.
    '''
    order = 0  # we start assuming that the best order is the first one
    nvar = len(projections[0])  # the number of variables corresponds with the length of the list describing one of the projections

    for i in range(nvar):
        # projections[order] is the projection that if chosen order we assume to be the best. All orders we can still choose from are equal to this one until this point
        try:
            if heuristic != 'greedy_sotd':
                new_var = greedy_choose_variable(projections[order][i], heuristic=heuristic)
            elif i < nvar-1:
                new_var = greedy_choose_variable([projections[ordering][i+1] for ordering in range(factorial(nvar)) if projections[ordering][i]== projections[order][i]], heuristic=heuristic)
            else:
                new_var = 0
        except IndexError:
            # The reason of this error is probably that the computation of the projection did not go further, in this case we return the current order
            return order

        if type(new_var) == str:
            return order
        order = order + factorial(nvar-i-1) * new_var  # the best order is updated with the new information
    return order  # the final best order is returned


def not_greedy_heuristic_guess(original_polynomials: list,
                               heuristic: str = "gmods"):
    '''
    This function is especialized in not greedy heuristics.
    All variables are picked from the original polynomials.
    '''
    order = 0  # we start assuming that the best order is the first one
    order_measure = get_order_measure(heuristic, if_tie=None)
    degrees_list, nvar = get_degree_list(original_polynomials, heuristic)
    variables = list(range(nvar))
    ordering = []

    while len(variables) != 0:
        best_vars = variables
        for measure in order_measure:
            best_vars = choose_variables_minimizing(degrees_list, measure='gmods', var_list=best_vars)
        random.shuffle(best_vars)
        # print('best vars shuffled', best_vars)
        ordering += best_vars
        variables = [var for var in variables if var not in ordering]
    assignment = {'[0, 1, 2]': 0, '[0, 2, 1]': 1,
                  '[1, 0, 2]': 2, '[1, 2, 0]': 3,
                  '[2, 0, 1]': 4, '[2, 1, 0]': 5,
                  }
    order = assignment[str(ordering)]
    # order = order + factorial(nvar-i-1) * new_var # the best order is updated with the new information
    return order  # the final best order is returned


def no_greedy_heuristic_guess(projections:list, heuristic:str="old_mods"):
    '''
    Looking at the same time at all the projections, 
    the no greedy heuristics make an ordering choice.
    '''
    if heuristic == "sotd":
        sotd_values = [sum([degree for level in projection for polynomial in level for monomial in polynomial for degree in monomial[:-1]]) for projection in projections]
        return min(range(len(sotd_values)), key=sotd_values.__getitem__) # returns the index with the smallest value in the list sotd_values
    elif heuristic in ["old_mods", "logmods", "mods", "acc_logmods"]:
        nvar = len(projections[0])
        pseudorderings = create_pseudorderings(nvar)
        relevant_degrees = [[[max([monomial[var] for monomial in polynomial]) for polynomial in level] for level,var in zip(projection,pseudordering)] for projection, pseudordering in zip(projections, pseudorderings)] # This returns a list of lists, each of those lists correspond to a projection. Those lists contain lists of the degrees of the polynomials in each level wrt the variable that will be projected after.
        heuristic_dict = {'old_mods':old_mods_guess, 'mods':mods_guess,'logmods':logmods_guess, 'acc_logmods':acc_logmods_guess}
        return heuristic_dict[heuristic](relevant_degrees)
    else:
        raise Exception("Heuristic "+heuristic+" not found.")
