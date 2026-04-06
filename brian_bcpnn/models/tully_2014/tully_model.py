from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("./")

from brian_bcpnn.models.tully_2014.tully_params import *
from brian_bcpnn.plot import trains, traces

defaultclock.dt = 0.1*ms

N_total = 3

min_num = 10e-6

dI = -0.4 * nA
stimulation_protocol = TimedArray(np.transpose([
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]), dt=100*ms)

eqs_rec = '''
# POSTSYNAPTIC (j) TRACES
dS_j/dt = -S_j/sim_dt : 1
dZ_j/dt = (S_j/(f_max*t_spike) - Z_j + epsilon)/tau_z : 1
dE_j/dt = (Z_j - E_j)/tau_e : 1
dP_j/dt = (K*(E_j-P_j))/tau_p : 1

# bias
beta = log(clip(P_j, min_num, inf)) : 1
I_beta = phi*beta : amp

# on-switch
b_on = stim_ta(t,i) : 1
dg_stim/dt = -g_stim/tau_AMPA : siemens
I_stim = b_on * g_stim * (V_m-E_ex) : amp

# total voltage
g_ex : siemens # summed over all excitatory synapses
g_inh : siemens # summed over all inhibitory synapses
dV_m/dt = (g_L*(V_m-E_L)+g_ex*(V_m-E_ex)+g_inh*(V_m-E_inh)+I_beta + I_stim)/-C_m : volt (unless refractory)
'''

REC = NeuronGroup(N_total, model=eqs_rec, method='euler', threshold='V_m>=V_th', reset='V_m=V_res', refractory=t_ref)

rec_syn_model = '''
# excitatory/inhibitory conductance
w_g = abs(w) * g_max : siemens # weighted maximum condunctance

dS_ex/dt = -S_ex/tau_ex : 1 (clock-driven) # excitatory conducting window
dalpha_ex/dt = (S_ex-alpha_ex)/tau_ex : 1  (clock-driven)
g_ex_post = b_ex * w_g * alpha_ex : siemens (summed)

dS_inh/dt = -S_inh/tau_inh : 1 (clock-driven) # inhibitory conducting window
dalpha_inh/dt = (S_inh-alpha_inh)/tau_inh : 1 (clock-driven)
g_inh_post = (1-b_ex) * w_g * alpha_inh : siemens (summed)

# PRESYNAPTIC (i) TRACES
dS_i/dt = -S_i/sim_dt : 1 (clock-driven)
dZ_i/dt = (S_i/(f_max*t_spike) - Z_i + epsilon)/tau_z : 1 (clock-driven)
dE_i/dt = (Z_i - E_i)/tau_e : 1 (clock-driven)
dP_i/dt = (K*(E_i-P_i))/tau_p : 1 (clock-driven)

# BCPNN synapse
dE_syn/dt = (Z_i*Z_j_post - E_syn)/tau_e : 1 (clock-driven)
dP_syn/dt = (K*(E_syn-P_syn))/tau_p : 1 (clock-driven)
w = log(clip(P_syn, min_num, inf)/clip(P_i*P_j_post, min_num, inf)) : 1 (constant over dt)
b_ex : int(w > 0) # 1 if synapse is excitatory, 0 if inhibitory
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
S_REC.connect(condition='i!=j')
S_REC.b_ex[:] = 'rand() > 0.2'
# print(S_REC.b_ex, 0 in S_REC.b_ex)

spikemon = SpikeMonitor(REC)
statemon = StateMonitor(REC, ['V', 'S_j', 'Z_j', 'E_j', 'P_j', 'beta', 'g_ex', 'g_inh', 'I_beta', 'I_on'], record=[0, 1])
synmon = StateMonitor(S_REC, ['w', 'Z_i', 'E_i', 'P_i', 'E_syn', 'P_syn'], record=True)

tfinal = 1000 * ms
run(tfinal)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (1, 3, 3, 3)})

# trains.get_full_train(ax, spikemon, N_total)
trains.compare_two_trains(ax1, spikemon, 0, 1)

traces.plot_z_traces(ax2, statemon, synmon, S_REC, 0, 1)

traces.plot_e_traces(ax3, statemon, synmon, S_REC, 0, 1)

traces.plot_p_traces(ax4, statemon, synmon, S_REC, 0, 1)

ax4.set_xticks(np.arange(0, tfinal / ms, 100))
ax4.set_xlabel("Time (ms)")

# plt.legend()
plt.show()