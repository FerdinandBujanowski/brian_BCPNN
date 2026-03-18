from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

def get_full_train(ax, spikemon, N, t_total, x_label=None, c='k', t_div=1*ms):
    ax.scatter(spikemon.t/t_div, spikemon.i[:], marker='_', color=c, s=10)
    ax.set_xlim(0, t_total/t_div)
    ax.set_ylim(0, N)
    ax.set_ylabel('# neuron')
    ax.set_yticks(np.arange(0, N, max(10, int(N/5))))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    
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


def get_neuron_frequency(spikemon:SpikeMonitor, neuron, t_stop, t_start=0*ms):
    spike_trains = spikemon.spike_trains()
    relevant_times = [t for t in spike_trains[neuron] if t >= t_start and t < t_stop]
    return len(relevant_times) / (t_stop - t_start)

# def get_event_frequency(eventmon, neuron, t_stop, t_start=0*ms):
#     return len(eventmon.event_trains()[neuron]) / (t_stop - t_start)

def get_spiking_histogram(ax, spikemon, N, t_stop, t_start=0*ms):
    freqs = [get_neuron_frequency(spikemon, i, t_stop, t_start)/Hz for i in range(N)]
    ax.hist(freqs, bins=5)
    return freqs