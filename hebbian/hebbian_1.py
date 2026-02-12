from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

defaultclock.dt = 1 * ms

# NETWORK PARAMETERS
N_input = 10 * 2
N_hyper = 10
N_mini = 5
N_hidden = N_hyper * N_mini

# GLOBAL PARAMETERS
delta_t = defaultclock.dt
F_MAX = 100 * hertz
mu_spk = delta_t * F_MAX
tau_m = 5 * ms

# Z/P TRACE PARAMETERS
tau_z = 20 * ms
tau_p = 5 * second
# tau_zi = tau_zj = 20 * ms
# tau_pi = tau_pj = tau_pij = 5 * second

# INPUT LAYER
eqs_input = '''
'''
INPUT = NeuronGroup(N_input, eqs_input)

# HIDDEN LAYER
eqs_hidden = '''
ds/dt = -s/delta_t : 1 # spike train
dz/dt = (s/mu_spk-z)/tau_z : 1
dp/dt = (z-p)/tau_p : 1
b = log(p) : 1
zw : 1 # sum of z_i * w_i_j of all synapses

I_ext : 1
dv/dt = b + zw + I_ext - v : 1
'''
HID = NeuronGroup(N_hidden, eqs_hidden, method='euler') # TODO add threshold and reset

ff_syn_eqs = '''
zw_post = z_pre * w : 1 (summed)
dpij/dt = (z_pre*z_post-pij)/tau_p : 1
w = log(pij/(p_pre*p_post)) : 1
'''

S_REC = Synapses(HID, HID, model=ff_syn_eqs, on_pre='s_pre=1', on_post='s_post=1')