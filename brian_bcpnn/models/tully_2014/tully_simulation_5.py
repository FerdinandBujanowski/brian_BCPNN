from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append("./")
from brian_bcpnn.networks import TullyNetwork
from brian_bcpnn.plot import composite, trains
from brian_bcpnn.utils.stim_utils import StimProtocol, ColumnCoords, StimTime
import brian_bcpnn.utils.stim_utils as stils
import brian_bcpnn.utils.synapse_utils as syls
from brian_bcpnn.models.tully_2014.tully_params import tully_equations, tully_namespace
# from activation_patterns import activation_lists


#NEW_TAU_P = 100*ms # As in Tully.  1000ms!!
NEW_TAU_P = 1000*ms
dt = 0.01 * ms
defaultclock.dt = dt
epsilon_n = 0.0033 # epsilon = f_min/f_max, a baseline firing rate
# before epsilon = 1 /(f_max * tau_p) = 0.0033
model_run_length = 500

# --------- SINGLE SPIKE PAIRS ------------

#i = 10*ms # spike timing interval
i = 5*ms

'''
start_scope()
tully_namespace['epsilon'] = epsilon_n
time_after=100*ms
model = TullyNetwork()
tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)
tully_namespace['tau_p'] = NEW_TAU_P # add in loop also
#tully_namespace['epsilon'] = epsilon_n
# model.namespace['tau_e'] = new_tau_e
weightmon = model.add_synmon(variables=['w'], record=True)
spikemon = model.add_spikemon()
#basmon = model.add_basmon()
biasmon = StateMonitor(source=model.REC, variables='beta', record=True)
model.add_monitor(biasmon, biasmon.name)
# TODO add statemon for bias
# w_before = model.S_REC.w[0]
model.run(5*ms)
w_before = model.S_REC.w[0] # S_REC is the synapse between i and j, so this is the synaptic strength aka the weight. Weight of the first synapse. 
t_total = 10*NEW_TAU_P

if i > 0:
    model.REC.V_m[0] = 0*mV # spike presyn
    model.run(abs(i))            # time between 
  #  model.REC.V_m[1] = 0*mV # spike postsyn
    model.REC.V_m[0] = 0*mV
    model.run(abs(i))
  #  model.REC.V_m[0] = 0*mV
    model.run(abs(i))
   # model.REC.V_m[0] = 0*mV
    model.run(abs(i))
   # model.REC.V_m[0] = 0*mV
    model.run(abs(i))
    #model.REC.V_m[0] = 0*mV
    #model.run(abs(i))
    model.run(30*ms)
    model.REC.V_m[1] = 0*mV
    #model.REC.V_m[0] = 0*mV
    #model.run(abs(i))
    #model.REC.V_m[0] = 0*mV''
else:
    model.REC.V_m[1] =0*mV  # spike presyn
    model.run(abs(i))       # time between 
    model.REC.V_m[0] = 0*mV # spike postsyn


time_after = time_after - (5*ms + abs(i))
model.run(time_after)
w_after = model.S_REC.w[0] # index at timestep of 100 ms 

model.run(model_run_length*ms)

print('spiking intervall', i, 'ms')
print('weight after 5 ms:', w_before)
print('weight after 100ms: ', w_after)
print('delta weight: ', (w_after - w_before))
model_run_length = model_run_length + 100 + 5 
'''

start_scope()

#P_syn_values = [15*epsilon_n**2, 20*epsilon_n**2, 30*epsilon_n**2, 50*epsilon_n**2]  
P_syn_values = [15*epsilon_n**2, 20*epsilon_n**2, 25*epsilon_n**2, 30*epsilon_n**2, 35*epsilon_n**2, 40*epsilon_n**2]
weight_traces = {}  # stores {P_syn: (t, w)} per run
#bias_traces = {}

for P_syn in P_syn_values:

    start_scope() 
    time_after=100*ms
    tully_namespace['epsilon'] = epsilon_n
    model = TullyNetwork()
   # tully_namespace['epsilon'] = epsilon_n
    tully_namespace['tau_p'] = NEW_TAU_P
    tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)

   
    eps = epsilon_n
    model.S_REC.set_states({
        'Z_i': eps, 'E_i': eps, 'P_i': 2*eps,
        'E_syn': eps**2, 'P_syn': P_syn  
    })

    weightmon = model.add_synmon(variables=['w'], record=True)
    spikemon = model.add_spikemon()
    #biasmon = StateMonitor(model.REC, 'beta', record=True)
    biasmon = StateMonitor(source=model.REC, variables='beta', record=True)
    model.add_monitor(biasmon, biasmon.name)

    # 
    model.run(5*ms)

    if i > 0:
        model.REC.V_m[1] = 0*mV
        model.run(abs(i))
        model.REC.V_m[1] = 0*mV
        model.run(abs(i))
        model.REC.V_m[1] = 0*mV
        model.run(abs(i))
    #   model.REC.V_m[0] = 0*mV
    #   model.run(abs(i))
    #    model.REC.V_m[0] = 0*mV
    #    model.run(abs(i))
    #    model.REC.V_m[0] = 0*mV
    #    model.run(abs(i))
    #    model.REC.V_m[0] = 0*mV
    #    model.run(abs(i))
    #   model.REC.V_m[0] = 0*mV
    #    model.run(abs(i))
    else:
        model.REC.V_m[1] = 0*mV
        model.run(abs(i))
        model.REC.V_m[0] = 0*mV

    time_after_run = (100*ms) - (5*ms + abs(i))
    model.run(time_after_run)
    model.run(model_run_length * ms)
    # -----------------------------------------

    w_before = weightmon.w[0][0]                                             
    w_after_100 = weightmon.w[0][np.searchsorted(weightmon.t, 100*ms)] 

    w_0 = float(np.log(P_syn) - np.log((10*epsilon_n) * (10*epsilon_n)))  # OBS CHANGE PI PJ MANUALLY 
    weight_traces[P_syn] = (
    weightmon.t/ms,
    weightmon.w[0].copy(),
    biasmon.beta[0].copy(),  # pre-synaptic bias
    biasmon.beta[1].copy(),  # post-synaptic bias
    w_0,
    w_before,
    w_after_100
)

for P_syn, (t, w, beta0, beta1, w_0, w_before, w_after_100) in weight_traces.items():
    print(f'w_0={w_0:.2f}, beta0 range: {beta0.min():.4f} to {beta0.max():.4f}')


# new code above

w_starts = [v[5] for v in weight_traces.values()]
w_ends   = [v[6] for v in weight_traces.values()]



fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})

# Raster plot - scatter of (time, neuron index)
ax1.set_ylabel('Neuron index')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Pre (0)', 'Post (1)'])
ax1.set_title('Spike raster')
ax1.scatter(spikemon.t[spikemon.i == 0]/ms, spikemon.i[spikemon.i == 0], 
            marker='|', s=1400, c='red', label='Pre-synaptic spike (0)', linewidths=2)
ax1.scatter(spikemon.t[spikemon.i == 1]/ms, spikemon.i[spikemon.i == 1], 
            marker='|', s=1400, c='blue', label='Post-synaptic spike (1)', linewidths=2)
ax1.set_ylabel('Spikes')

ax2.scatter(w_starts, w_ends, zorder=3)
ax2.plot([min(w_starts), max(w_starts)],
         [min(w_starts), max(w_starts)],
         'k--', linewidth=1, label='y = x (no change)')  # diagonal reference line
ax2.set_xlabel('Initial weight (w₀)')
ax2.set_ylabel('Weight after 100 ms')
ax2.set_title('Weight change')
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.axhline(y=0, color='grey', linewidth=1.5, linestyle='--')
#ax2.axvline(x=0, color='black', linewidth=1.5, linestyle='--')
#ax4.set_xlim(-2,5)

plt.tight_layout()
plt.show()