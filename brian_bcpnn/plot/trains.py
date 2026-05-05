from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append("./")
import brian_bcpnn.utils.synapse_utils as sils
from brian_bcpnn.utils.stim_utils import StimTime, PatternList, Pattern
import brian_bcpnn.utils.spike_utils as spils

pattern_cmap = mpl.colormaps['viridis']
# cmap((1+i)/(len(pt_dict)+1))

def get_full_train(ax, spikemon, N, t_total, x_label=None, c='k', t_div=1*ms, fr=0, to=None):
    max_N = N if to is None else to
    t_array, i_array = [], []
    for t, i in zip(spikemon.t, spikemon.i):
        if i >= fr and i < max_N:
            t_array.append(t)
            i_array.append(i)
    
    # spikemon.t, spikemon.i
    ax.scatter(t_array/t_div, i_array, marker='_', color=c, s=10)
    ax.set_xlim(0, t_total/t_div)
    ax.set_ylim(fr-0.5, max_N)
    ax.set_ylabel('# neuron')
    ax.set_yticks(list(range(fr, max_N, 180)))
    ax.grid(axis='y')
    
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def compare_two_trains(ax, spikemon, n_a, n_b, x_label=None, c_a='r', c_b='b', t_div=ms):
    spike_trains = spikemon.spike_trains()
    train_a = spike_trains[n_a]
    train_b = spike_trains[n_b]
    for train, c in zip([train_a, train_b], [c_a, c_b]):
        t=[0]
        s=[0]
        for i in train:
            i_ms = i/t_div
            for t_i, s_i in zip([i_ms, i_ms, i_ms], [0, 1, 0]):
                t.append(t_i)
                s.append(s_i)
        ax.plot(t, s, c=c, alpha=0.5)

    # ax.set_xlim(0, len(spikemon.t/t_div))
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def get_spiking_histogram(ax, spikemon, N, t_stop, t_start=0*ms):
    freqs = [spils.get_neuron_frequency(spikemon, i, t_stop, t_start)/Hz for i in range(N)]
    if ax is not None:
        ax.hist(freqs, bins=5)
    return freqs

def get_active_freqs_per_batch(
        ax,
        spikemon:SpikeMonitor, n_pattern, 
        N_M, N_pyr,
        patterns:PatternList, pt_dict:dict[str,list[StimTime]]
):
    coord_list = patterns.patterns[n_pattern].coord_list
    batch_times = pt_dict[list(pt_dict.keys())[n_pattern]] # list[StimTime]
    
    pattern_freqs = np.zeros(shape=(len(batch_times), len(coord_list)*N_pyr))
    # loop throught the start/stop times of each batch's stimulation
    for n_batch, batch_time in enumerate(batch_times):
        # loop through all HC-MC coordinates of the chosen pattern
        for i_c, coords in enumerate(coord_list):
            # loop through all individual neurons of current minicolumn
            for current_pyr in range(N_pyr):
                # add current neuron frequency to np array of freqs
                pattern_freqs[n_batch, i_c*N_pyr+current_pyr] = spils.get_neuron_frequency(
                    spikemon=spikemon, neuron=sils.get_first_pyr(coords.HC, coords.MC, N_M, N_pyr),
                    t_start=batch_time.t_start, t_stop=batch_time.t_end
                )/Hz
    ax.boxplot(pattern_freqs.T)

# TODO rework this to work with both distorted and partial cues
def plot_all_minicolumn_activations(ax, model, spikemon, full_patterns:PatternList, stims, t_total, dt, t_div=second, show_cued=False):
    # cmap = mpl.colormaps['viridis']
    # cmap((1+i)/(len(pt_dict)+1))

    # pattern_1_train = trains.get_pattern_population_train(spikemon, model, pattern_list.patterns[0], t_total, defaultclock.dt)
    # pattern_1_freqs = trains.get_firing_rate_estimate(pattern_1_train, N_M*N_pyr, 0.1, t_total/ms, kernel_size=15)#
    ax.set_ylabel('Firing Rate / Hz')

    time_array = np.arange(0, t_total-dt, dt)/t_div
    for i_h in range(model.N_H):
        for i_m in range(model.N_M):
            occ_patterns = []
            ls = '-' if show_cued else ':'
            c = 'r'
            for i_pattern, pattern in enumerate(full_patterns.patterns):
                for coords in pattern.coord_list:
                    if i_h == coords.HC and i_m == coords.MC:
                        occ_patterns.append(i_pattern)
                        for stim in stims:
                            if coords == stim.coords:
                                ls = '-'
            if len(occ_patterns) == 1:
                c = pattern_cmap((1+i_pattern)/(len(full_patterns.patterns)+1))
            if len(occ_patterns) > 0:
                mc_train = spils.get_minicolumn_population_train(spikemon, model, i_h, i_m, t_total, dt)
                fr_estimate = spils.get_firing_rate_estimate(mc_train, model.N_pyr, dt/ms, t_total/ms, kernel_size=20)
                ax.plot(time_array, fr_estimate, color=c, alpha=0.5, ls=ls)