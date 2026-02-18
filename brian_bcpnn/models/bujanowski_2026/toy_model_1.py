# Toy BCPNN model based on Tully LIF neurons

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("./")

from brian_bcpnn.models.tully_2014.parameters import * # change this to my own param file asp
import brian_bcpnn.helper as hlp
from brian_bcpnn.plot import trains, traces, synapses

# NETWORK PARAMETERS
N_hyper = 8
N_mini = 8
N_total = N_hyper * N_mini

defaultclock.dt = sim_dt
min_num = 10e-6
dI = -0.5 * nA

# RECURRENT HYPER-MINI-COLUMN LAYER
eqs_rec = '''
# POSTSYNAPTIC (j) TRACES
dS_j/dt = -S_j/sim_dt : 1
dZ_j/dt = (S_j/(f_max*t_spike) - Z_j + epsilon*second)/tau_z : 1
dE_j/dt = (Z_j - E_j)/tau_e : 1
dP_j/dt = (K*(E_j-P_j))/tau_p : 1

# bias
beta = log(clip(P_j, min_num, inf)) : 1
I_beta = phi*beta : amp

# on-switch
b_on : 1 # no stimulation protocol here
I_on = b_on * dI : amp

# total voltage
g_ex : siemens # summed over all excitatory synapses
g_inh : siemens # summed over all inhibitory synapses
g_lat_inh : siemens # summed over all LATERAL inhibitory synapses
dV/dt = (g_L*(V-E_L)+g_ex*(V-E_ex)+(g_inh+g_lat_inh)*(V-E_inh)+I_beta+I_on)/-C_m : volt (unless refractory)
'''

REC = NeuronGroup(N_total, model=eqs_rec, method='euler', threshold='V>=V_th', reset='V=V_res', refractory=t_ref)
REC.V = -70 * mV

# BCPNN SYNAPSES
rec_syn_model = '''
# excitatory/inhibitory conductance
b_ex : 1 # 1 if synapse is excitatory, 0 if inhibitory
w_g = w * g_max : siemens # weighted maximum condunctance

dS_ex/dt = -S_ex/tau_ex : 1 (clock-driven) # excitatory conducting window
dalpha_ex/dt = (S_ex-alpha_ex)/tau_ex : 1  (clock-driven)
g_ex_post = b_ex * w_g * alpha_ex : siemens (summed)

dS_inh/dt = -S_inh/tau_inh : 1 (clock-driven) # inhibitory conducting window
dalpha_inh/dt = (S_inh-alpha_inh)/tau_inh : 1 (clock-driven)
g_inh_post = (1-b_ex) * w_g * alpha_inh : siemens (summed)

# PRESYNAPTIC (i) TRACES
dS_i/dt = -S_i/sim_dt : 1 (clock-driven)
dZ_i/dt = (S_i/(f_max*t_spike) - Z_i + epsilon*second)/tau_z : 1 (clock-driven)
dE_i/dt = (Z_i - E_i)/tau_e : 1 (clock-driven)
dP_i/dt = (K*(E_i-P_i))/tau_p : 1 (clock-driven)

# BCPNN synapse
dE_syn/dt = (Z_i*Z_j_post - E_syn)/tau_e : 1 (clock-driven)
dP_syn/dt = (K*(E_syn-P_syn))/tau_p : 1 (clock-driven)
w = log(clip(P_syn, min_num, inf)/clip(P_i*P_j_post, min_num, inf)) : 1 (constant over dt)
'''

rec_syn_on_pre = '''
S_i = 1
S_ex = 1
S_inh = 1
'''
rec_syn_on_post = '''
S_j_post = 1
'''

S_REC = Synapses(REC, REC, model=rec_syn_model, on_pre=rec_syn_on_pre, on_post=rec_syn_on_post, method='euler', delay=d)
S_REC.connect(condition='i!=j') # fully connected recurrent layer
S_REC.b_ex[:] = 1 # = 'rand() > 0.2'

# LATERAL INHIBITORY SYNAPSES
lat_inh_model = '''
dS_inh/dt = -S_inh/tau_inh : 1 (clock-driven)
dalpha_inh/dt = (S_inh-alpha_inh)/tau_inh : 1 (clock-driven)
g_lat_inh_post = g_max * alpha_inh : siemens (summed)
'''
lat_inh_on_pre = '''
S_inh = 1
'''

S_LAT = Synapses(REC, REC, model=lat_inh_model, on_pre=lat_inh_on_pre, method='euler')
source_inh, target_inh = hlp.get_inh_synapses(N_hyper, N_mini)
S_LAT.connect(i=source_inh, j=target_inh)

fig, ax = plt.subplots()
im = synapses.plot_connectivity(ax, S_LAT, N_total)
fig.colorbar(im, ax=ax)
plt.show()

# MONITORS
spikemon = SpikeMonitor(REC)
rec_statemon = hlp.get_BCPNN_statemon(REC, record=True)
bcpnn_synmon = hlp.get_BCPNN_synmon(S_REC, record=True)

t_sample = 150 * ms
n_samples = 8
n_batches = 4
tfinal = n_batches * n_samples * t_sample

for i_batch in range(n_batches):
    for i_sample in tqdm(range(n_samples)):
        REC.b_on = [1 if i % 8 == i_sample else 0 for i in range(N_total)]
        run(t_sample)

fig, ax = plt.subplots()
trains.get_full_train(ax, spikemon, N_total, tfinal)

plt.show()

fig, ax_array = plt.subplots(2, 4)
for i, ax in enumerate(np.ndarray.flatten(ax_array)):
    current_t = int(np.round((i/8 * (tfinal/defaultclock.dt-1))))
    ax.set_title(f't={current_t*defaultclock.dt}')
    im = synapses.plot_weights_at_t(ax, bcpnn_synmon, S_REC, N_total, current_t)
    fig.colorbar(im, ax=ax)

plt.show()