# Updated version of tully_model.py, this time with i and j traces saved at the Synapse level

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from plot import trains

defaultclock.dt = sim_dt

N_hyper = 8
N_mini = 8
N_total = N_hyper * N_mini

min_num = 10e-6

eqs_rec = '''
# traces
dS_j/dt = -S_j/sim_dt : 1
dZ_j/dt = (S_j/(f_max*t_spike) - Z_j + epsilon*second)/tau_z : 1
dE_j/dt = (Z_j - E_j)/tau_e : 1
dP_j/dt = (K*(E_j-P_j))/tau_p : 1

# bias
beta = log(clip(P_j, min_num, inf)) : 1
I_beta = phi*beta : amp

# on-switch
b_on : 1
I_on = -0.1 * nA : amp

# total voltage
g_ex : siemens # summed over all excitatory synapses
g_inh : siemens # summed over all inhibitory synapses
dV/dt = ((1-b_on)*g_L*(V-E_L)+g_ex*(V-E_ex)+g_inh*(V-E_inh)+I_beta + b_on*I_on)/-C_m : volt (unless refractory)
'''

REC = NeuronGroup(N_total, model=eqs_rec, method='euler', threshold='V>=V_th', reset='V=V_res', refractory=t_ref)
REC.V = -70 * mV
REC.b_on[0] = 1
REC.b_on[8] = 1

rec_syn_model = '''
# excitatory/inhibitory conductance
b_ex : 1 # 1 if synapse is excitatory, 0 if inhibitory
w_g = w * g_max : siemens # weighted maximum condunctance

dS_ex/dt = -S_ex/tau_ex : 1 (clock-driven) # excitatory conducting window
dalpha_ex/dt = (S_ex-alpha_ex)/tau_ex : 1  (clock-driven)
g_ex_post = b_ex * w_g * alpha_ex : siemens (summed)

dS_inh/dt = -S_inh/tau_inh : 1 (clock-driven) # inhibitory conducting window
dalpha_inh/dt = (S_inh-alpha_inh)/tau_inh : 1 (clock-driven)
g_inh_post = (1-b_ex) * w_g * S_inh : siemens (summed)

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
S_REC.connect(condition='i!=j')
S_REC.b_ex[:] = 'rand() > 0.2'
# print(S_REC.b_ex, 0 in S_REC.b_ex)

first_targets = []
for source, target in zip(S_REC.i, S_REC.j):
    if source == 0:
        first_targets.append(target)

spikemon = SpikeMonitor(REC)
statemon = StateMonitor(REC, ['V', 'S_j', 'Z_j', 'E_j', 'P_j', 'beta', 'g_ex', 'g_inh', 'I_beta', 'I_on'], record=[0, 1])
synmon = StateMonitor(S_REC, ['w', 'P_syn', 'E_syn'], record=True)

tfinal = 1000 * ms
run(tfinal)


fig, (ax, ax_voltage) = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (1, 4)})

# trains.get_full_train(ax, spikemon, N_total)
trains.compare_two_trains(ax, spikemon, 0, 7)

ax_voltage.plot(statemon.t / ms, statemon.E_j[0], # np.clip(statemon.v[0], -np.inf, 30),
               color='k')

ax_voltage.set_xticks(np.arange(0, tfinal / ms, 100))
ax_voltage.spines['right'].set_visible(False)
ax_voltage.spines['top'].set_visible(False)
ax_voltage.set_xlabel("time, ms")

plt.show()

plt.plot(statemon.t/ms, synmon[S_REC[0, 8]].w[0])
plt.show()