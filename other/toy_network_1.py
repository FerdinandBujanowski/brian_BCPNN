from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from brian_bcpnn.helper import get_inh_synapses

#prefs.codegen.target = 'numpy'
#prefs.codegen.loop_invariant_optimisations = False
#np.seterr(all='raise')

defaultclock.dt = 1 * ms

# NETWORK PARAMETERS
N_hyper = 8
N_mini = 8
N_total = N_hyper * N_mini

# GLOBAL PARAMETERS
delta_t = defaultclock.dt
F_MAX = 100 * hertz
mu_spk = delta_t * F_MAX
tau_m = 10 * ms
V_t = 20 * mV
V_m = 0 * mV
min_num = 1e-6

R = 4*Mohm
I_spike = 0.1*uA

# Z/P TRACE PARAMETERS
tau_z = 20 * ms
tau_p = 5 * second

# RECURRENT LAYER
eqs_rec = '''
test : 1

ds_i/dt = -s_i/delta_t : 1 # presynaptic spike train
ds_j/dt = -s_j/delta_t : 1 # postsynaptic spike train

dz_i/dt = (s_i/mu_spk-z_i)/tau_z : 1 # presynaptic z-trace
dz_j/dt = (s_j/mu_spk-z_j)/tau_z : 1 # postsynaptic z-trace

dp_i/dt = (z_i-p_i)/tau_p : 1 # presynaptic p-trace
dp_j/dt = (z_j-p_j)/tau_p : 1 # postsynaptic p-trace

b = log(clip(p_j, min_num, inf)) : 1
zw : 1 # sum of z_i * w_i_j of all synapses

dI/dt = -I/(20*ms) : amp
I_ext : amp
dv/dt = (-v+R*(I+I_ext)+(b+zw)*mV)/tau_m : volt
'''

REC = NeuronGroup(N_total, eqs_rec, method='euler', threshold='v>=V_t', reset='v=V_m')
REC.v = V_m
for i_n in [0, 8, 16, 24, 32, 40, 48, 56]:
    REC.I_ext[i_n] = 0.1 * uA

rec_syn_model = '''
w = log(clip(pij, min_num, inf)/clip(p_i_pre, min_num, inf)*clip(p_j_post, min_num, inf)) : 1 (constant over dt)

dpij/dt = (z_i_pre*z_j_post-pij)/tau_p : 1 (clock-driven)

zw_post = z_i_pre * w : 1 (summed)
'''

S_REC = Synapses(REC, REC, model=rec_syn_model, on_pre='I_post+=w*I_spike;s_i_pre+=1;s_j_post+=1', method='euler')
S_REC.connect(condition='i!=j')

source_inh, target_inh = get_inh_synapses(N_hyper, N_mini)
S_INH = Synapses(REC, REC, model='w=-20:1', on_pre='I_post+=w*I_spike', method='euler')
S_INH.connect(i=source_inh, j=target_inh)

spikemon = SpikeMonitor(REC)
statemon = StateMonitor(REC, ['v', 's_i', 's_j', 'z_i', 'z_j', 'p_i', 'p_j', 'b', 'zw', 'test'], record=0)
synmon = StateMonitor(S_REC, ['w'], record=True)

tfinal = 1000 * ms
run(tfinal)

# print(synmon.w.shape)
# print(synmon.w[0])

fig, (ax, ax_voltage) = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': (3, 1)})

ax.scatter(spikemon.t / ms, spikemon.i[:], marker="_", color="k", s=10)
ax.set_xlim(0, tfinal / ms)
ax.set_ylim(0, N_total)
ax.set_ylabel("neuron number")
ax.set_yticks(np.arange(0, N_total, 100))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax_voltage.plot(statemon.t / ms, statemon.b[0], # np.clip(statemon.v[0], -np.inf, 30),
               color='k')

ax_voltage.set_xticks(np.arange(0, tfinal / ms, 100))
ax_voltage.spines['right'].set_visible(False)
ax_voltage.spines['top'].set_visible(False)
ax_voltage.set_xlabel("time, ms")

plt.show()