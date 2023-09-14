"""Make some plots"""
import os
import numpy as np
from numpy import sort
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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
