from brian2 import *
import matplotlib.pyplot as plt
import numpy as np



def get_full_train(ax, spikemon, N, x_label=None):
    ax.scatter(spikemon.t/ms, spikemon.i[:], marker='_', color='k', s=10)
    ax.set_xlim(0, len(spikemon.t/ms))
    ax.set_ylim(0, N)
    ax.set_ylabel('# neuron')
    ax.set_yticks(np.arange(0, N, 100))
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def compare_two_trains(ax, spikemon, n_a, n_b, x_label=None, c_a='r', c_b='b'):
    spike_trains = spikemon.spike_trains()
    train_a = spike_trains[n_a]
    train_b = spike_trains[n_b]
    for train, c in zip([train_a, train_b], [c_a, c_b]):
        t=[0]
        s=[0]
        for i in train:
            i_ms = i/ms
            for t_i, s_i in zip([i_ms, i_ms, i_ms], [0, 1, 0]):
                t.append(t_i)
                s.append(s_i)
        ax.plot(t, s, c=c, alpha=0.5)

    ax.set_xlim(0, len(spikemon.t/ms))
    ax.set_ylim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax
