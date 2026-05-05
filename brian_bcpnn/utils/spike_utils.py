from brian2 import *
import numpy as np
from scipy import ndimage

sys.path.append("./")
from brian_bcpnn.utils.stim_utils import Pattern, PatternList, StimTime
import brian_bcpnn.utils.synapse_utils as syls

def get_neuron_frequency(spikemon:SpikeMonitor, neuron, t_stop, t_start=0*ms):
    spike_trains = spikemon.spike_trains()
    relevant_times = [t for t in spike_trains[neuron] if t >= t_start and t < t_stop]
    return len(relevant_times) / (t_stop - t_start)

def get_minicolumn_frequency(model, spikemon:SpikeMonitor, HC, MC, t_stop, t_start=0*ms):
    first_index = syls.get_first_pyr(HC, MC, model.N_M, model.N_pyr)
    spike_trains = spikemon.spike_trains()
    spike_counts = np.zeros(shape=(model.N_pyr,))
    for i_neuron in range(first_index, first_index+model.N_pyr):
        for t in spike_trains[i_neuron]:
            if t >= t_start and t < t_stop:
                spike_counts[i_neuron-first_index] += 1
    
    t_interval = t_stop-t_start
    total_freq = sum(spike_counts) / t_interval
    mean_freq = np.mean(spike_counts) / t_interval
    std_freq = np.std(spike_counts) / t_interval

    return total_freq, mean_freq, std_freq

# def get_event_frequency(eventmon, neuron, t_stop, t_start=0*ms):
#     return len(eventmon.event_trains()[neuron]) / (t_stop - t_start)

def get_discrete_spike_trains(spikemon, neuron_indices, t_total, dt):
    spike_times = spikemon.spike_trains()

    n_steps = int(t_total/dt)
    discrete_trains = np.zeros(shape=(len(neuron_indices), n_steps))

    for i, neuron_index in enumerate(neuron_indices):
        for t in spike_times[neuron_index]:
            discrete_trains[i][int(t/dt)] = 1.
    return discrete_trains

def get_minicolumn_population_train(spikemon, model, H, M, t_total, dt):
    start_neuron = H*model.N_M*model.N_pyr + M*model.N_pyr
    index_range = list(range(start_neuron, start_neuron+model.N_pyr))

    return np.sum(get_discrete_spike_trains(spikemon, index_range, t_total, dt), axis=0)

def get_pattern_population_train(spikemon, model, pattern:Pattern, t_total, dt):

    all_indices = []
    for coords in pattern.coord_list:
        start_neuron = coords.HC*model.N_M*model.N_pyr + coords.MC*model.N_pyr
        all_indices = all_indices + list(range(start_neuron, start_neuron+model.N_pyr))
    
    return np.sum(get_discrete_spike_trains(spikemon, all_indices, t_total, dt), axis=0)

def get_firing_rate_estimate(spiketrain, population_size, spikebinsize, phase_duration, kernel_size=10):
    # computes population firing rate across time and then divides by the population size
    # to return the avg firing rate per neuron
    if len(spiketrain) == 0:
        return np.zeros(int(phase_duration / spikebinsize))

    # Calculate the number of bins explicitly
    num_bins = int(phase_duration / spikebinsize)

    # Generate the histogram with the exact number of bins
    # discrete, _ = np.histogram(spiketrain, bins=num_bins, range=(0, phase_duration))

    signal = ndimage.gaussian_filter1d(spiketrain, sigma=kernel_size / (2.35 * spikebinsize), mode='constant') # kernelwidth ~50 ms, fwhm=2.35*sigma

    # No need for padding since we already have the correct number of bins
    fr = signal * (1000. / spikebinsize) # spikes per bin * (1 second / bin size) to transform to Herz

    return fr / population_size

def eval_pattern_activation(model, spikemon:SpikeMonitor, pattern:Pattern, t_stop, t_start=0*ms, pattern_bound=20*Hz, non_pattern_bound=10*Hz):
    pattern_freqs = []
    non_pattern_freqs = []
    for H in range(model.N_H):
        for M in range(model.N_M):
            part_of_pattern = False
            _, mean_freq, _ = get_minicolumn_frequency(model, spikemon, H, M, t_stop, t_start)
            for coords in pattern.coord_list:
                if H == coords.HC and M == coords.MC:
                    part_of_pattern = True
                    pattern_freqs.append(mean_freq)
            if not part_of_pattern:
                non_pattern_freqs.append(mean_freq)
    
    
    all_pattern_above = bool(np.all(np.array(pattern_freqs) >= pattern_bound/Hz))
    # print(pattern_freqs, all_pattern_above)
    # all_nonpattern_below = bool(np.all(np.array(non_pattern_freqs) < non_pattern_bound/Hz))
    # print(non_pattern_freqs, all_nonpattern_below)
    return all_pattern_above

def eval_pattern_completion(model, spikemon, pattern_list:PatternList, pt_dict:dict[str,list[StimTime]], t_isi:Quantity):
    completion_list = []

    pt_dict_keys = list(pt_dict.keys())
    for i, pattern in enumerate(pattern_list.patterns):
        stim_times = pt_dict[pt_dict_keys[i]]
        for stim_time in stim_times:
            evaluation = eval_pattern_activation(model, spikemon, pattern, t_start=stim_time.t_end, t_stop=stim_time.t_end+0.5*t_isi)
            completion_list.append((i, evaluation))
    
    return completion_list
