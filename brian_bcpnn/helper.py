from brian2 import StateMonitor
from numpy.random import random

bcpnn_layer_variables = ['V', 'S_j', 'Z_j', 'E_j', 'P_j', 'beta', 'g_ex', 'g_inh', 'I_beta', 'I_on']
bcpnn_synapse_variables = ['w', 'Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn', 'S_ex', 'w_g']

def get_inh_synapses(n_hyper, n_mini):

    source = []
    target = []

    for i_hyper in range(n_hyper):
        hyper_offset = i_hyper * n_hyper
        for source_mini in range(hyper_offset, hyper_offset+n_mini):
            for target_mini in range(hyper_offset, hyper_offset+n_mini):
                if source_mini != target_mini:
                    source.append(source_mini)
                    target.append(target_mini)
    
    return source, target

def get_neuron_coords(i, N_mini, N_pyr):
    return i // (N_mini*N_pyr), (i % (N_mini*N_pyr)) // N_pyr

def get_rec_synapses(N_hyper, N_mini, N_pyr, cp_same_mini, cp_same_hyper, cp_diff_hyper):
    source = []
    target = []
    N = N_hyper * N_mini * N_pyr
    
    for i_pre in range(N):
        pre_h, pre_m = get_neuron_coords(i_pre, N_mini, N_pyr)
        for i_post in range(N):
            if i_pre != i_post:
                post_h, post_m = get_neuron_coords(i_post, N_mini, N_pyr)
                if (
                    (pre_m == post_m and random() < cp_same_mini) or
                    (pre_h == post_h and random() < cp_same_hyper) or
                    (random() < cp_diff_hyper)
                ):
                    source.append(i_pre)
                    target.append(i_post)

    return source, target

def get_basket_synapses(N_hyper, N_mini, N_pyr, N_basket, cp_PB, cp_BP):
    source_P = []
    target_B = []
    source_B = []
    target_P = []

    N = N_hyper * N_mini * N_pyr
    N_basket_total = N_hyper * N_mini * N_basket
    N_basket_per_hyper = N_mini * N_basket

    for i_pre in range(N):
        pre_h, pre_m = get_neuron_coords(i_pre, N_mini, N_pyr)
        basket_offset = N_basket_per_hyper*pre_h + N_basket*pre_m

        for i_basket in range(basket_offset + N_basket):
            if random() < cp_PB:
                source_P.append(i_pre)
                target_B.append(i_basket)
            if random() < cp_BP:
                source_B.append(i_basket)
                target_P.append(i_pre)
    return (source_P, target_B, source_B, target_P)


def get_BCPNN_statemon(neurons, record):
    return StateMonitor(neurons, bcpnn_layer_variables, record=record)

def get_BCPNN_synmon(synapse, record):
    return StateMonitor(synapse, bcpnn_synapse_variables, record=record)

def get_BCPNN_weight_synmon(synapse, record):
    return StateMonitor(synapse, ['w'], record=record)