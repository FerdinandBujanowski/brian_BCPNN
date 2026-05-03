from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

# this is for recurrently connected networks

def plot_weights_at_t(ax, synmon, syn, N, t):
    syn_mat = np.zeros(shape=(N, N))
    for i, j in zip(syn.i, syn.j):
        syn_mat[i, j] = synmon[syn[i,j]].w[0][t]

    im = ax.imshow(syn_mat, cmap='viridis')
    return im

def plot_weights(ax, syn, N_i, N_j=None):
    syn_mat = np.zeros(shape=(N_i, N_j if N_j is not None else N_i))
    for i, (s, t) in enumerate(zip(syn.i, syn.j)):
        syn_mat[s, t] = syn.w[i]
    
    im = ax.imshow(syn_mat, cmap='viridis')
    return im

def plot_connectivity(ax, syn, N_i, N_j=None, colors:list[tuple[list,list,tuple[int,int,int]]]=None, aspect='auto'):
    syn_mat = np.zeros(shape=(N_i, N_j if N_j is not None else N_i, 3))
    for s, t in zip(syn.i, syn.j):
        if colors is None:
            syn_mat[s,t] = [1, 1, 1]
        else:
            b_in_list = False
            for (s_list,t_list,(r,g,b)) in colors:
                if s in s_list and t in t_list:
                    b_in_list = True
                    syn_mat[s,t] = [r, g, b]
            if not b_in_list:
                syn_mat[s,t] = [1, 1, 1]
    
    ax.set_ylabel('presynaptic neuron')
    ax.set_xlabel('postsynaptic neuron')    
    im = ax.imshow(syn_mat, aspect=aspect, interpolation='none')
    return im

def hist_presyn_count(ax, syn, N_j):
    counts = np.zeros(shape=(N_j,), dtype=int32)
    for i in syn.i:
        counts[i] += 1

    ax.set_xlabel('# of presyn. connections')
    ax.set_ylabel('Count')
    ax.hist(counts)

def hist_syn_weights(ax, syn):
    ax.set_xlabel('Synaptic Weight')
    ax.set_ylabel('Count')
    ax.hist(syn.w/1,bins=10)

# TODO seperate logic and plot code, move one to synapse_utils and delete the other if not enough code
def plot_weight_trajectory(
        ax, synmon:StateMonitor, monitored_indices:list[int],
        t_div=ms, plot_std=True, c='b', label=None
):
    t_step = len(synmon.t)
    t_array = synmon.t/t_div
    weights = np.zeros(shape=(len(monitored_indices), t_step))

    for i_w, i_syn in enumerate(monitored_indices):
        weights[i_w,:] = synmon[i_syn].w
    
    w_mean = np.mean(weights, axis=0)
    w_std = np.std(weights, axis=0)
    ax.plot(t_array, w_mean, label=label, c=c)
    if plot_std:
        n_std = 1.96
        ax.fill_between(t_array, w_mean-n_std*w_std, w_mean+n_std*w_std, alpha=0.3, color=c)
    
    ax.grid()
    ax.set_ylabel('synaptic weight')
    ax.set_xlabel(f'Time/{t_div}')

def plot_weight_matrix_averages(ax, model):
    length = model.N_H * model.N_M
    weight_mat = np.zeros(shape=(model.N, model.N))
    avg_weight_mat = np.zeros(shape=(length, length))
    PYR = model.N_pyr

    for i_syn, (s, t) in enumerate(zip(model.S_REC.i, model.S_REC.j)):
        weight_mat[s,t] = model.S_REC.w[i_syn]
    for i in range(length):
        for j in range(length):
            if i != j:
                weight_slice = weight_mat[i*PYR:(i+1)*PYR, j*PYR:(j+1)*PYR]
                avg_weight_mat[i,j] = np.sum(weight_slice) / max(np.count_nonzero(weight_slice), 1)
            else:
                avg_weight_mat[i,j] = model.namespace['intra_hc_intra_mc']
    ax.set_ylabel('presynaptic minicolumn')
    ax.set_xlabel('postsynaptic minicolumn')    
    im = ax.imshow(avg_weight_mat, aspect='auto', interpolation='none')
    return im