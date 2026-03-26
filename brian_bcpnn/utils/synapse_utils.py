from brian2 import *
from numpy.random import random

def get_neuron_coords(i, N_mini, N_pyr):
    return i // (N_mini*N_pyr), (i % (N_mini*N_pyr)) // N_pyr

def get_first_pyr(current_H, current_M, N_M, N_pyr):
    return current_H*N_M*N_pyr + current_M*N_pyr

def get_rec_synapses(N_hyper, N_mini, N_pyr, cp_same_mini, cp_same_hyper, cp_diff_hyper):
    source = []
    target = []
    N = N_hyper * N_mini * N_pyr
    
    for i_pre in range(N):
        pre_h, pre_m = get_neuron_coords(i_pre, N_mini, N_pyr)
        for i_post in range(N):
            b_append = False
            if i_pre != i_post:
                post_h, post_m = get_neuron_coords(i_post, N_mini, N_pyr)
                if pre_m == post_m and pre_h == post_h:
                    b_append = random() < cp_same_mini
                elif pre_m != post_m and pre_h == post_h:
                    b_append = random() < cp_same_hyper
                else:
                    b_append = random() < cp_diff_hyper
                    
            if b_append:
                source.append(i_pre)
                target.append(i_post)

    return source, target

def get_basket_synapses(N_hyper, N_mini, N_pyr, N_basket, cp_PB, cp_BP, symmetry=False):
    source_P = []
    target_B = []
    source_B = []
    target_P = []

    N = N_hyper * N_mini * N_pyr
    N_basket_per_hyper = N_mini * N_basket

    for i_pre in range(N):
        pre_h, _ = get_neuron_coords(i_pre, N_mini, N_pyr)
        basket_offset = N_basket_per_hyper*pre_h

        for i_basket in range(basket_offset, basket_offset+N_basket_per_hyper):
            if random() < cp_PB:
                source_P.append(i_pre)
                target_B.append(i_basket)
            if not symmetry and random() < cp_BP:
                source_B.append(i_basket)
                target_P.append(i_pre)

    if symmetry:
        return (source_P, target_B, target_B, source_P)
    return (source_P, target_B, source_B, target_P)

# for current_H in range(5):
#     for current_M in range(2):
#         print(get_first_pyr(current_H, current_M, 2, 30))