"""Make some plots"""
import os
import pickle
import numpy as np
from numpy import sort
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from find_filename import find_timings_lists
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

fontsize = 15
desired_font = {'fontname': 'monospace'}
matplotlib.rcParams.update({'font.size': fontsize})


def survival_plot(timings: dict, plot_name="survival_plot"):
    """Receive a dictionary where the keys are the name
    of the methos and the timings that took for each of
    the problems"""
    color = cm.rainbow(np.linspace(0, 1, len(timings)+1))
    # color[4]=[0.8,0.8,0.2,1]
    # color[3]=[0.65,0.42,0.42,1]
    # color[2]=[0.00,1,0.5,1]
    # color = ['0','0.5','0','0.5','0','0.5']
    style = ['--'] * len(timings)
    dashes = [(1, 0), (5, 1), (5, 1, 1, 1), (2, 1, 2, 1), (1, 1), (5, 5)]\
        + [(1, 0)] * len(timings)

    for method, c, s, d in zip(timings, color, style, dashes):
        not_timeout_timings = [timing for timing in timings[method]
                               if timing != 30 and timing != 60]
        sorted_timings = sort(not_timeout_timings)
        accumulative_timings = [sum(sorted_timings[:i])
                                for i in range(len(sorted_timings))]
        # plotting
        plt.plot(accumulative_timings, list(range(len(accumulative_timings))),
                 s, color=c, label=method, dashes=d)
    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('No. problems finished', fontsize=fontsize)
    plt.legend(prop={'family': 'monospace', 'size': fontsize-2},
               loc='lower right')
    figure_location = os.path.join(os.path.dirname(__file__), 'Art',
                                   f'{plot_name}.png')
    plt.savefig(figure_location)
    plt.cla()


def create_adversarial_plot(
    model1='RF',
    model2='RFR'
):
    '''
    This function creates an adversarial plot comparing the desired models.
    '''

    timings_lists_filename = find_timings_lists(model1)
    with open(timings_lists_filename, 'rb') as timings_lists_file:
        rawtimings1 = pickle.load(timings_lists_file)
    timings1 = [80 if timing == 60 else timing for timing in rawtimings1]

    timings_lists_filename = find_timings_lists(model2)
    with open(timings_lists_filename, 'rb') as timings_lists_file:
        rawtimings2 = pickle.load(timings_lists_file)
    timings2 = [80 if timing == 60 else timing for timing in rawtimings2]
    plot, ax = plt.subplots(1, 1)

    # Set number of ticks for x-axis
    ticks = list(np.arange(0, 90, 10))
    ticks.pop(-2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Set ticks labels for x-axis
    ticks_labels = ticks
    ticks_labels[-1] = 'Timeout'
    ax.set_xticklabels(ticks_labels, fontsize=fontsize)
    ax.set_yticklabels(ticks_labels, rotation='vertical', fontsize=fontsize)

    # plotting
    ax.plot(timings1, timings2, '.')
    ax.plot([0, 90], [0, 90], '-')

    # creating labels
    plt.xlabel(model1, **desired_font, fontsize=fontsize-2)
    plt.ylabel(model2, **desired_font, fontsize=fontsize-2)

    plt.title('Adversarial plot comparing ' + model1 + ' and ' + model2)
    figure_location = os.path.join(os.path.dirname(__file__), 'Art',
                                   'adversarial_plot_' + model1
                                   + '_vs_' + model2 + '.png')
    plt.savefig(figure_location)
    plt.cla()


# create_adversarial_plot()
