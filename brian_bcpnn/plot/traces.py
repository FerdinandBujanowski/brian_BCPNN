from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

def plot_z_traces(ax, statemon, synmon, syn, i, j, x_label=None, c_i='r', c_j='b'):
    ax.plot(statemon.t/ms, synmon[syn[i, j]].Z_i[0], c=c_i, label='presynaptic')
    ax.plot(statemon.t/ms, statemon.Z_j[j], c=c_j, label='postsynaptic') # i,j could mean different things for syn and state if record != True
    ax.set_ylabel('Z-traces')
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def plot_e_traces(ax, statemon, synmon, syn, i, j, x_label=None, c_i='r', c_j='b', c_ij='k'):
    ax.plot(statemon.t/ms, synmon[syn[i, j]].E_i[0], c=c_i, label='presynaptic')
    ax.plot(statemon.t/ms, statemon.E_j[j], c=c_j, label='postsynaptic')
    ax.plot(statemon.t/ms, synmon[syn[i,j]].E_syn[0], c=c_ij, label='synaptic')
    ax.set_ylabel('E-traces')
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax

def plot_p_traces(ax, statemon, synmon, syn, i, j, x_label=None, c_i='r', c_j='b', c_ij='k'):
    ax.plot(statemon.t/ms, synmon[syn[i, j]].P_i[0], c=c_i, label='presynaptic')
    ax.plot(statemon.t/ms, statemon.P_j[j], c=c_j, label='postsynaptic')
    ax.plot(statemon.t/ms, synmon[syn[i,j]].P_syn[0], c=c_ij, label='synaptic')
    ax.set_ylabel('P-traces')
    if x_label is not None:
        ax.set_xlabel(x_label)

    return ax