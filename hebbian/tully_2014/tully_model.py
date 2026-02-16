from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from parameters import *

defaultclock.dt = sim_dt

N_hyper = 8
N_mini = 8
N_total = N_hyper * N_mini

min_num = 10e-6

eqs_rec = '''
# traces
dS/dt = -S/sim_dt : 1
dZ/dt = (S/(f_max*t_spike) - Z + epsilon*second)/tau_z : 1
dE/dt = (Z - E)/tau_e : 1
dP/dt = (K*(E-P))/tau_p : 1

# bias
beta = log(clip(P, min_num, inf)) : 1
I_beta = phi*beta : amp

# on-switch
b_on : 1
I_on = (-C_m*f_max*(V_th-V_res)*sim_dt/1000)/ms-g_L*(V-E_L) : amp

# total voltage
# g_ext : siemens
# I_ext : amp
g_ex : siemens # summed over all excitatory synapses
g_inh : siemens # summed over all inhibitory synapses
dV/dt = (g_L*(V-E_L)+(g_ex)*(V-E_ex)+g_inh*(V-E_inh)+I_beta + b_on*I_on)/-C_m : volt (unless refractory)
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
dS_inh/dt = -S_inh/tau_inh : 1 (clock-driven) # inhibitory conducting window
g_ex_post = b_ex * w_g * S_ex : siemens (summed)
g_inh_post = (1-b_ex) * w_g * S_inh : siemens (summed)

# BCPNN synapse
dE_syn/dt = (Z_pre*Z_post - E_syn)/tau_e : 1 (clock-driven)
dP_syn/dt = (K*(E_syn-P_syn))/tau_p : 1 (clock-driven)
w = log(clip(P_syn, min_num, inf)/clip(P_pre, min_num, inf)*clip(P_post, min_num, inf)) : 1 (constant over dt)
'''

rec_syn_on_pre = '''
S_pre = 1
S_ex = 1
S_inh = 1
'''
rec_syn_on_post = '''
S_post = 1
'''

S_REC = Synapses(REC, REC, model=rec_syn_model, on_pre=rec_syn_on_pre, on_post=rec_syn_on_post, method='euler', delay=d)
S_REC.connect(condition='i!=j')
S_REC.b_ex = 'rand() > 0.2'

first_targets = []
for source, target in zip(S_REC.i, S_REC.j):
    if source == 0:
        first_targets.append(target)

spikemon = SpikeMonitor(REC)
statemon = StateMonitor(REC, ['V', 'S', 'Z', 'E', 'P', 'beta', 'g_ex', 'g_inh', 'I_beta', 'b_on'], record=[0, 1])


tfinal = 1000 * ms
run(tfinal)

fig, (ax, ax_voltage) = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (3, 1)})

ax.scatter(spikemon.t / ms, spikemon.i[:], marker="_", color="k", s=10)
ax.set_xlim(0, tfinal / ms)
ax.set_ylim(0, N_total)
ax.set_ylabel("neuron number")
ax.set_yticks(np.arange(0, N_total, 100))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax_voltage.plot(statemon.t / ms, statemon.V[0]/mV, # np.clip(statemon.v[0], -np.inf, 30),
               color='k')

ax_voltage.set_xticks(np.arange(0, tfinal / ms, 100))
ax_voltage.spines['right'].set_visible(False)
ax_voltage.spines['top'].set_visible(False)
ax_voltage.set_xlabel("time, ms")

plt.show()