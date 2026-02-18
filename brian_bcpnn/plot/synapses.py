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

def plot_connectivity(ax, syn, N):
    syn_mat = np.zeros(shape=(N, N))
    for i, j in zip(syn.i, syn.j):
        syn_mat[i, j] = 1
    im = ax.imshow(syn_mat, cmap='grey')
    return im