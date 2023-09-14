'''
This file contains the functions to create the graphs comparing the heuristics.
'''

from pydoc_data import topics
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pickle
import os
import numpy as np
from numpy import sort
from numpy import Inf

from .heuristics_guess import choose_order_given_projections
from .heuristic_tools import get_dataset, substract_two_timings, finding_time_limit, compute_markups, compute_real_timings

import matplotlib
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fontsize = 15
desired_font = {'fontname':'monospace'}
matplotlib.rcParams.update({'font.size': fontsize})

folder_figures = os.path.join(os.path.dirname(__file__), '..','Art')

######################################################
###LEARN TO USE PNG https://riptutorial.com/matplotlib/example/10066/saving-and-exporting-plots-that-use-tex#:~:text=In%20order%20to%20include%20plots,text%20in%20the%20final%20document.&text=Plots%20in%20matplotlib%20can%20be,macro%20package%20to%20display%20graphics.
########################################################


def create_survival_plot(
    heuristics = ['virtual_best', 'gmods', 'mods', 'brown', 'sotd', 'greedy_sotd'],
    minimum_time_to_consider=0,
    rep=10
    ):
    '''This function creates a survival plot comparing the desired heuristics.'''

    dataset = get_dataset(without_repetition=True, minimum_time_to_consider=minimum_time_to_consider)
    projections, targets, timings, heuristics_costs, ncells = dataset

    color = cm.rainbow(np.linspace(0, 1, len(heuristics)+1))
    # color[4]=[0.8,0.8,0.2,1]
    color[3]=[0.65,0.42,0.42,1]
    color[2]=[0.00,1,0.5,1]
    #color = ['0','0.5','0','0.5','0','0.5']
    style = ['--','--','--','--','--','--']
    dashes = [(1,0),(5,1),(5,1,1,1),(2,1,2,1),(1,1),(5,5)]
    
    for heuristic, c, s, d in zip(heuristics,color, style, dashes):
        many_sorted_timings = []
        for i in range(rep):
            if heuristic=='virtual_best':
                rawtimings = [timing[target] for timing, target in zip(timings, targets)]
            else:
                guesses = [choose_order_given_projections(projection, method=heuristic) for projection in projections]
                rawtimings = [timing[guess] for timing, guess in zip(timings, guesses)]
            sorted_timings = sort([timing for timing, all_orders_timing in zip(rawtimings,timings) if type(timing)!=str and timing<finding_time_limit(all_orders_timing)]) # This eliminates not only strings but also choices that together with the heuristic cost got over the time limit.
            many_sorted_timings.append(sorted_timings)
        avg_sorted_timings = combine_many_sorted_timings(many_sorted_timings, penalization=120)
        accumulative_timings = [sum(avg_sorted_timings[:i]) for i in range(len(avg_sorted_timings))]

        #plotting
        if heuristic==36:
            heuristic = "T1"
        elif heuristic == 67:
            heuristic = "T2"
        plt.plot(accumulative_timings, list(range(len(accumulative_timings))), s, color=c, label=heuristic, dashes=d)
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('No. problems finished', fontsize=fontsize)
    plt.legend(prop={'family':'monospace', 'size':fontsize-2}, loc='lower right')
    figure_location = os.path.join(folder_figures,'survival_plot__without_repetition__min_time_'+str(minimum_time_to_consider)+'.png')
    plt.savefig(figure_location)
    plt.cla()

def combine_many_sorted_timings(many_sorted_timings, penalization=120):
    avg_len = round(sum([len(sorted_timings) for sorted_timings in many_sorted_timings])/ len(many_sorted_timings))
    #many_sorted_timings_longenough = [sorted_timings+[penalization]*avg_len for sorted_timings in many_sorted_timings]
    avg_sorted_timings = [sum([penalization if i>=len(st) else st[i] for st in many_sorted_timings])/len( many_sorted_timings) for i in range(avg_len)]
    return avg_sorted_timings

def create_adversarial_plot(
    heuristic1 = 'gmods',
    heuristic2 = 'avegavegdeg'
    ):
    '''This function creates an adversarial plot comparing the desired heuristics.'''

    dataset = get_dataset(without_repetition=True, minimum_time_to_consider=0)
    # we always want all examples here
    projections, _, timings, heuristics_costs, ncells = dataset

    guesses1 = [choose_order_given_projections(projection, method=heuristic1) for projection in projections]
    rawtimings1 = [timing[guess] for timing, guess in zip(timings, guesses1)]
    timings1 = [timing if type(timing)!=str and timing<finding_time_limit(all_orders_timing) else 80 for timing, all_orders_timing in zip(rawtimings1,timings)] # This eliminates not only strings but also choices that together with the heuristic cost got over the time limit.


    guesses2 = [choose_order_given_projections(projection, method=heuristic2) for projection in projections]
    rawtimings2 = [timing[guess] for timing, guess in zip(timings, guesses2)]
    timings2 = [timing if type(timing)!=str and timing<finding_time_limit(all_orders_timing) else 80 for timing, all_orders_timing in zip(rawtimings2,timings)] # This eliminates not only strings but also choices that together with the heuristic cost got over the time limit.

    plot, ax = plt.subplots(1,1) 

    # Set number of ticks for x-axis
    ticks = list(np.arange(0,90,10))
    ticks.pop(-2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Set ticks labels for x-axis
    ticks_labels = ticks
    ticks_labels[-1] = 'Timeout'
    ax.set_xticklabels(ticks_labels, fontsize=fontsize) #
    ax.set_yticklabels(ticks_labels, rotation='vertical', fontsize=fontsize)

    #plotting
    ax.plot(timings1, timings2, '.')
    ax.plot([0,60],[0,60],'-')

    #creating labels
    plt.xlabel(heuristic1, **desired_font, fontsize=fontsize-2)
    plt.ylabel(heuristic2, **desired_font, fontsize=fontsize-2) 

    # plt.title('Adversarial plot comparing '+heuristic1+' and '+heuristic2)
    figure_location = os.path.join(folder_figures, 'adversarial_plot_'+heuristic1+'_vs_'+heuristic2+'.png') #, '..\\','Art','adversarial_plot_'+heuristic1+'_vs_'+heuristic2+'.png')
    plt.savefig(figure_location)
    plt.cla()


def plot_comparing_mods_and_gmods():

    dataset = get_dataset(without_repetition=True, minimum_time_to_consider=0)
    # we always want all examples here
    projections, targets, timings, heuristics_costs, ncells = dataset

    guesses_mods = [choose_order_given_projections(projection, method='mods') for projection in projections]
    timings_mods = [timing[guess] for timing, guess in zip(timings,guesses_mods)]
    heuristic_costs = [heuristics_cost[target] for heuristics_cost, target in zip(heuristics_costs, targets)]


    guesses_gmods = [choose_order_given_projections(projection, method='gmods') for projection in projections]
    timings_gmods = [timing[guess] for timing, guess in zip(timings,guesses_gmods)]

    virtual_best_timings = [timing[target] for timing, target in zip(timings,targets)]
    timings_diff = [substract_two_timings(gmods_time,mods_time) for gmods_time, mods_time in zip(timings_gmods, timings_mods)]

    # Create Plot

    fig, ax1 = plt.subplots() 
    top = 30
    nticks = 7
    ax1.set_yticks(np.linspace(-top, top, nticks))
    
    ax1.set_xlabel('Time taken by virtual_best ordering', fontsize=fontsize) 
    ax1.set_ylabel('Ordering difference', color = 'red', fontsize=fontsize) 
    ax1.plot(virtual_best_timings, timings_diff, '.', color = 'red') 
    ax1.tick_params(axis ='y', labelcolor = 'red') 
    
    # Adding Twin Axes

    ax2 = ax1.twinx() 
    
    ax2.set_ylim([-top, top])
    ax2.set_yticks(np.linspace(-top, top, nticks))
    ax2.set_ylabel('Heuristic cost', color = 'blue', fontsize=fontsize) 
    ax2.plot(virtual_best_timings, heuristic_costs, '.', color = 'blue', fontsize=fontsize) 
    ax2.tick_params(axis ='y', labelcolor = 'blue')


    # plt.show()
    figure_location = os.path.join(folder_figures, 'plot_comparing_mods_and_gmods.png') #, '..\\','Art','adversarial_plot_'+heuristic1+'_vs_'+heuristic2+'.png')
    plt.savefig(figure_location)
    


def plot_comparing_mods_and_gmods2(max_penalization_if_not_finished=Inf):

    dataset = get_dataset(without_repetition=True, minimum_time_to_consider=0)
    # we always want all examples here
    projections, targets, timings, heuristics_costs, ncells = dataset

    virtual_best_timings = [timing[target] for timing, target in zip(timings,targets)]

    guesses_mods = [choose_order_given_projections(projection, method='mods') for projection in projections]
    choice_timings_including_str_mods = [timing[guess] for timing, guess in zip(timings, guesses_mods)]
    choice_timings_mods = compute_real_timings(timings, choice_timings_including_str_mods, virtual_best_timings, max_penalization_if_not_finished=max_penalization_if_not_finished) 
    markups_mods = compute_markups(virtual_best_timings, choice_timings_mods)

    guesses_gmods = [choose_order_given_projections(projection, method='gmods') for projection in projections]
    choice_timings_including_str_gmods = [timing[guess] for timing, guess in zip(timings, guesses_gmods)]
    choice_timings_gmods = compute_real_timings(timings, choice_timings_including_str_gmods, virtual_best_timings, max_penalization_if_not_finished=max_penalization_if_not_finished) 
    markups_gmods = compute_markups(virtual_best_timings, choice_timings_gmods)

    

    # Create Plot

    
    plt.plot(virtual_best_timings, markups_mods, '.', color = 'red') 
    plt.plot(virtual_best_timings, markups_gmods, '.', color = 'blue') 
    plt.xlabel('virtual_best time', fontsize=fontsize)
    plt.ylabel('Markups', fontsize=fontsize)
    

    # plt.show()
    figure_location = os.path.join(folder_figures, 'plot_comparing_mods_and_gmods2.png') #, '..\\','Art','adversarial_plot_'+heuristic1+'_vs_'+heuristic2+'.png')
    plt.savefig(figure_location)


def create_difficulty_histogram():
    '''This function create a histogram showing the distribution of difficulty among the problems.'''

    dataset = get_dataset(without_repetition=True, minimum_time_to_consider=0)
    projections, targets, timings, heuristics_costs, ncells = dataset

    op_timings = [timing[target] for timing, target in zip(timings, targets)]
    plt.yscale('log')
    plt.hist(op_timings, bins=list(range(0,65,5)))
    plt.xlabel('seconds in optimal ordering', fontsize=fontsize)
    plt.ylabel('number of problems', fontsize=fontsize)
    # plt.show()
    figure_location = os.path.join(folder_figures, 'histogram_difficulty.png') #, '..\\','Art','adversarial_plot_'+heuristic1+'_vs_'+heuristic2+'.png')
    plt.savefig(figure_location)



# create_difficulty_histogram()