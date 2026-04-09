from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append("./")
import brian_bcpnn.utils.synapse_utils as sils
from brian_bcpnn.utils.stim_utils import StimTime, PatternList

def get_full_train(ax, spikemon, N, t_total, x_label=None, c='k', t_div=1*ms, fr=0, to=None):
    max_N = N if to is None else to
    t_array, i_array = [], []
    for t, i in zip(spikemon.t, spikemon.i):
        if i >= fr and i < max_N:
            t_array.append(t)
            i_array.append(i)
    
    ax.scatter(t_array/t_div, i_array, marker='_', color=c, s=10)
    ax.set_xlim(0, t_total/t_div)
    ax.set_ylim(fr-0.5, max_N)
    ax.set_ylabel('# neuron')
    ax.set_yticks(np.arange(fr, max_N, max(10, int(max_N/5))))
    
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def compare_two_trains(ax, spikemon, n_a, n_b, x_label=None, c_a='r', c_b='b', t_div=ms):
    spike_trains = spikemon.spike_trains()
    train_a = spike_trains[n_a] # exact spike times
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

def get_neuron_frequency(spikemon:SpikeMonitor, neuron, t_stop, t_start=0*ms):
    spike_trains = spikemon.spike_trains()
    relevant_times = [t for t in spike_trains[neuron] if t >= t_start and t < t_stop]
    return len(relevant_times) / (t_stop - t_start)

# def get_event_frequency(eventmon, neuron, t_stop, t_start=0*ms):
#     return len(eventmon.event_trains()[neuron]) / (t_stop - t_start)

def get_spiking_histogram(ax, spikemon, N, t_stop, t_start=0*ms):
    freqs = [get_neuron_frequency(spikemon, i, t_stop, t_start)/Hz for i in range(N)]
    if ax is not None:
        ax.hist(freqs, bins=5)
    return freqs

def sliding_window_freq(ax, spikemon, N, t_stop, t_start=0*ms, window_size=100*ms, step_size=50*ms, t_div=second):
    total_window = t_stop - t_start
    n_steps = int((total_window-window_size)/step_size+1)
    freqs = np.zeros(shape=(n_steps, N))
    x_time = []
    for step in tqdm(range(n_steps)):
        current_start = t_start + step*step_size
        x_time.append(current_start)
        current_stop = current_start + window_size
        freqs[step,:] = get_spiking_histogram(ax=None, spikemon=spikemon, N=N, t_start=current_start, t_stop=current_stop)

    freqs_mean = np.mean(freqs, axis=1)
    # freqs_std = np.std(freqs, axis=1)
    color = 'b'
    ax.plot(x_time, freqs_mean, c=color, label='mean')
    # n_std = 1.96
    # ax.fill_between(x_time, freqs_mean-n_std*freqs_std, freqs_mean+n_std*freqs_std, alpha=0.3, color=color)
    ax.set_xlabel(f'Time/{t_div}')
    ax.set_ylabel('Firing Frequency')
    ax.legend()

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
                pattern_freqs[n_batch, i_c*N_pyr+current_pyr] = get_neuron_frequency(
                    spikemon=spikemon, neuron=sils.get_first_pyr(coords.HC, coords.MC, N_M, N_pyr),
                    t_start=batch_time.t_start, t_stop=batch_time.t_end
                )/Hz
    ax.boxplot(pattern_freqs.T)