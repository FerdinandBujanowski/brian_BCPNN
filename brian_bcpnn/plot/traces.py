from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

def plot_z_traces(ax, statemon, i, j, x_label=None, c_i='r', c_j='b', t_div=ms):
    ax.plot(statemon.t/t_div, statemon[i].Z, c=c_i, label='presynaptic')
    ax.plot(statemon.t/t_div, statemon[j].Z, c=c_j, label='postsynaptic') # i,j could mean different things for syn and state if record != True
    ax.set_ylabel('Z-traces')
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def plot_e_traces(ax, statemon, synmon, i_syn, i, j, x_label=None, c_i='r', c_j='b', c_ij='k', t_div=ms):
    ax.plot(statemon.t/t_div, statemon[i].E, c=c_i, label='presynaptic')
    ax.plot(statemon.t/t_div, statemon[j].E, c=c_j, label='postsynaptic')
    ax.plot(statemon.t/t_div, synmon.E_syn[i_syn], c=c_ij, label='synaptic')
    ax.set_ylabel('E-traces')
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def plot_p_traces(ax, statemon, synmon, i_syn, i, j, x_label=None, c_i='r', c_j='b', c_ij='k', t_div=ms):
    ax.plot(statemon.t/t_div, statemon[j].P, c=c_j, label='postsynaptic')
    ax.plot(statemon.t/t_div, statemon[i].P, c=c_i, label='presynaptic')
    ax.plot(statemon.t/t_div, synmon.P_syn[i_syn], c=c_ij, label='synaptic')
    ax.set_ylabel('P-traces')
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax