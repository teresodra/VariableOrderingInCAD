'''
This folder contains all the details necessary for the 
different heuristics to make their choices.
'''

import os
import numpy as np
from math import log
import itertools
import random
from .heuristic_tools import multiplyList, all_combinations, minimum_indices, aveg_of_not_zero, paper_all_pos


def choose_variables_minimizing(degrees_list, measure='gmods', var_list=''):
    '''Given a list the degrees of polynomials returns the list of variables that minimise the measure desired'''
    if measure != 'greedy_sotd':
        nvar = len(degrees_list[0][0])  # the number of variables will be the same everywhere, we check the first monomial of the first polynomial
    else:
        nvar = len(degrees_list[0][0][0])
    if var_list == '':  # if the value is the default one
        var_list = range(nvar)

    if measure == 'gmods':
        sum_degree_polys = [sum([max([monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degree_polys)] # var_list is filtered
    if measure == 'ali_aveg':
        av_degree_polys_with_var = [aveg_of_not_zero([max([monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about.
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(av_degree_polys_with_var)] # var_list is filtered
    elif measure == 'greedy_logmods':
        sum_degrees_overall_polys = [sum([log(max([1]+[monomial[var] for monomial in polynomial])) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about.
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)]
    elif measure == 'brown1':
        max_degrees_polywise = [max([max([monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the maximum degree in the polynomials is computed.
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(max_degrees_polywise)]
    elif measure == 'brown2':
        max_degrees_polywise = [max([max([0]+[monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the maximum degree in the polynomials is computed.

        degrees_of_monomials_with_max_degrees = [max([max([0]+[sum(monomial) for monomial in polynomial if monomial[var]==max_degrees_polywise[var]]) for polynomial in degrees_list]) for var in var_list] # for each variable, the maximum degree in the polynomials is computed.
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(degrees_of_monomials_with_max_degrees)]
    elif measure == 'brown3':
        number_appearances = [sum([sum([np.sign(monomial[var]) for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # the number of monomials in which the variables appear is counted
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(number_appearances)]
    # elif measure == 'avegmaxsvdeg':
    #     sum_degrees_overall_polys = [np.average([max([sum(monomial) for monomial in polynomial if monomial[var]>0]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
    #     return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    # elif measure == 'maxsumsvdeg':
    #     sum_degrees_overall_polys = [max([sum([sum(monomial) for monomial in polynomial if monomial[var]>0]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
    #     return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    # elif measure == 'avegsumsvdeg':
    #     sum_degrees_overall_polys = [np.average([sum([sum(monomial) for monomial in polynomial if monomial[var]>0]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
    #     return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    # elif measure == 'avegsumdeg':
    #     sum_degrees_overall_polys = [np.average([sum([monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
    #     return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    elif measure == 'avegavegdeg':
        aveg_degrees_overall_polys = [np.average([np.average([monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(aveg_degrees_overall_polys)] # var_list is filtered

    # elif measure == 'maxsumdeg':
    #     sum_degrees_overall_polys = [max([sum([monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
    #     return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    elif measure == 'sumsignsumdeg':
        sum_degrees_overall_polys = [np.sum(np.sign([np.sum([monomial[var] for monomial in polynomial]) for polynomial in degrees_list])) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    elif measure == 'sumsumdeg':
        sum_degrees_overall_polys = [sum([sum([monomial[var] for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    # elif measure == 'avegvegsigndeg':
    #     sum_degrees_overall_polys = [np.average([np.average([np.sign(monomial[var]) for monomial in polynomial]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
    #     return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    # elif measure == 'sumsumsvdeg':
    #     sum_degrees_overall_polys = [sum([sum([sum(monomial) for monomial in polynomial if monomial[var]>0]) for polynomial in degrees_list]) for var in var_list] # for each variable, the total degree of each polynomial is computed. Then for each variable this values are added because is what we really care about. 
    #     return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_degrees_overall_polys)] # var_list is filtered
    elif measure == 'greedy_sotd':
        sum_total_degrees = [sum([sum(monomial) for polynomial in possible_proj_set for monomial in polynomial]) for possible_proj_set in degrees_list]
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(sum_total_degrees)] # var_list is filtered
    elif measure == 'random':
        return [random.choice(var_list)]
    elif measure == 'first':
        return [var_list[0]]
    elif measure == 'last':
        return [var_list[-1]]
    elif type(measure)==str:
        if measure[-5:]=='svdeg':
            measure = measure[:-5]
            monomial_numbers = [[[sum(monomial) for monomial in polynomial if monomial[var]>0] for polynomial in degrees_list] for var in var_list]
        elif measure[-7:]=='signdeg':
            measure = measure[:-7]
            monomial_numbers = [[[np.sign(monomial[var]) for monomial in polynomial] for polynomial in degrees_list] for var in var_list]
        elif measure[-3:]== 'deg':
            measure = measure[:-3]
            monomial_numbers = [[[monomial[var] for monomial in polynomial] for polynomial in degrees_list] for var in var_list]
        else:
            raise Exception(measure+" is not a valid measure")

        if measure[-3:] == 'sum':
            measure = measure[:-3]
            polynomial_numbers = [[sum(monomial_numbers_in_poly) for monomial_numbers_in_poly in var_monomial_numbers] for var_monomial_numbers in monomial_numbers]
        elif measure[-3:] == 'max':
            measure = measure[:-3]
            polynomial_numbers = [[max(monomial_numbers_in_poly) if len(monomial_numbers_in_poly)>0 else 0 for monomial_numbers_in_poly in var_monomial_numbers] for var_monomial_numbers in monomial_numbers]
        elif measure[-4:] == 'aveg':
            measure = measure[:-4]
            polynomial_numbers = [[np.average(monomial_numbers_in_poly) for monomial_numbers_in_poly in var_monomial_numbers] for var_monomial_numbers in monomial_numbers]
        else:
            raise Exception("Not a valid measure - maybe add the possibility of sign here")

        if measure == 'sum':
            set_numbers = [sum(var_polynomial_numbers) for var_polynomial_numbers in polynomial_numbers]
        elif measure == 'max':
            set_numbers = [max(var_polynomial_numbers) if len(var_polynomial_numbers)>0 else 0 for var_polynomial_numbers in polynomial_numbers]
        elif measure == 'aveg':
            set_numbers = [np.average(var_polynomial_numbers) for var_polynomial_numbers in polynomial_numbers]
        else:
            raise Exception("Not a valid measure")
        return [var_list[i] for i in range(len(var_list)) if i in minimum_indices(set_numbers)]  # var_list is filtered


def get_order_measure(heuristic, if_tie='random'):
    if heuristic == 'brown':
        order_measure = ['brown1', 'brown2', 'brown3', if_tie]
    elif heuristic == 'T1':
        order_measure = ['gmods', 'avegavegdeg', 'sumsumdeg']
    elif type(heuristic) == int:
        order_measure = list(paper_all_pos[heuristic])+[if_tie]
    else:
        order_measure = [heuristic, if_tie]
    return order_measure


def get_degree_list(poly_list, heuristic):
    if heuristic != 'greedy_sotd':
        degrees_list = [[monomial[:-1] for monomial in polynomial] for polynomial in poly_list] # the same list without the coefficients
        nvar = len(degrees_list[0][0])  # the number of variables will be the same everywhere, we check the first monomial of the first polynomial
    else:
        degrees_list = [[[monomial[:-1] for monomial in polynomial] for polynomial in polys] for polys in poly_list] # the same list without the coefficients
        nvar = len(degrees_list[0][0][0])
    # if degrees_list == []:  # idk why this happens but we just return this sentence
    #     return "The list given is empty"
    return degrees_list, nvar


def greedy_choose_variable(poly_list, heuristic='gmods'):
    '''Given a list of polynomials returns the variable that the gmods heuristic would choose to project next'''

    order_measure = get_order_measure(heuristic, if_tie='random')
    degrees_list, nvar = get_degree_list(poly_list, heuristic)
    best_vars = range(nvar)
    n_random_choice = 1
    while len(best_vars) > 1:
        measure = order_measure.pop(0)
        if measure == 'random':
            # if we reach random we save how many variables are left
            n_random_choice = len(best_vars)
        best_vars = choose_variables_minimizing(degrees_list, measure=measure, var_list=best_vars)
    # The following three lines are just used to answer a question from the reviewers
    if nvar == 3 and (heuristic == 'gmods' or heuristic == 36 or heuristic == 'brown'):
        file_random_name = os.path.join(os.path.dirname(__file__), '..', 'Datasets', f"{heuristic}_random_choices.txt")
        with open(file_random_name, 'a') as f:
            f.write(f"{n_random_choice}, ")
    return best_vars[0]


##
# Rules for expensive heuristics
##

def old_mods_guess(mrd):#mrd->old_mods_relevant_degrees
    '''Computes the best order according to the old_mods heuristic (multiplication of relative degrees).'''
    old_mods_values = [multiplyList([sum([degree for degree in level_mrd if degree!=0]) for level_mrd in proj_mrd]) for proj_mrd in mrd]
    return min(range(len(old_mods_values)), key=old_mods_values.__getitem__) # returns the index with the smallest value in the list old_mods_values


def logmods_guess(mrd):
    '''Computes the best order according to the logmods heuristic (multiplication of the logarithm of relative degrees).'''
    logmods_values = [multiplyList([sum([log(degree) for degree in level_mrd if degree!=0]) for level_mrd in proj_mrd]) for proj_mrd in mrd]
    return min(range(len(logmods_values)), key=logmods_values.__getitem__) # returns the index with the smallest value in the list logmods_values


def mods_guess(mrd):
    '''Computes the best ordering minimizing the maximum number of cells in the final CAD.'''
    mods_values = [multiplyList([1+2*sum([degree for degree in level_mrd if degree!=0]) for level_mrd in proj_mrd]) for proj_mrd in mrd]
    return min(range(len(mods_values)), key=mods_values.__getitem__) # returns the index with the smallest value in the list old_mods_values


def super_mods_guess(mrd):
    '''Computes the best ordering minimizing the maximum number of cells in all the CADs needed to build the final CAD.'''
    mods_values = [sum([multiplyList([1+2*sum([degree for degree in level_mrd if degree!=0]) for level_mrd in proj_mrd[:i+1]]) for i in range(len(proj_mrd))])for proj_mrd in mrd]
    return min(range(len(mods_values)), key=mods_values.__getitem__) # returns the index with the smallest value in the list old_mods_values


def acc_logmods_guess(mrd):
    '''Computes the best order according to the logmods heuristic (multiplication of the logarithm of relative degrees).'''
    acc_logmods_values = [multiplyList([1+2*sum([log(degree) for degree in level_mrd if degree!=0]) for level_mrd in proj_mrd]) for proj_mrd in mrd]
    return min(range(len(acc_logmods_values)), key=acc_logmods_values.__getitem__) # returns the index with the smallest value in the list logmods_values
