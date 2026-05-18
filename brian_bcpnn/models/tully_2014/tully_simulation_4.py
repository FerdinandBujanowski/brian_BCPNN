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
epsilon_n = 0.0033 # epsilon = f_min/f_max, a baseline firing rate, Anders said  0.0033
# before epsilon = 1 /(f_max * tau_p) = 0.0033
model_run_length = 500
# 1 /(f_max * tau_p) = 1/(30 * 1 000)


# --------- SINGLE SPIKE PAIRS ------------

#i = 10*ms # spike timing interval
i = 5*ms


start_scope()
tully_namespace['epsilon'] = epsilon_n
time_after=100*ms
model = TullyNetwork()
tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)
tully_namespace['tau_p'] = NEW_TAU_P # add in loop also
#tully_namespace['epsilon'] = epsilon_n
# model.namespace['tau_e'] = new_tau_e

eps = epsilon_n
model.REC.set_states({
    'Z_j': eps, 'E_j': eps, 'P_j': eps
})

model.S_REC.set_states({
    'Z_i': eps, 'E_i': eps, 'P_i': eps,
    'E_syn': eps**2, 'P_syn': 5*eps**2  
})

weightmon = model.add_synmon(variables=['w'], record=True)
spikemon = model.add_spikemon()
#basmon = model.add_basmon()
biasmon = StateMonitor(source=model.REC, variables='beta', record=True)
model.add_monitor(biasmon, biasmon.name)
# TODO add statemon for bias
# w_before = model.S_REC.w[0]
model.run(5*ms)
w_before = model.S_REC.w[0] # S_REC is the synapse between i and j, so this is the synaptic strength aka the weight. Weight of the first synapse. 
w_before1 = weightmon.w[0][0]
#t_total = 10*NEW_TAU_P

if i > 0:
    model.REC.V_m[0] = 0*mV # spike presyn
    model.run(abs(i))            # time between 
  #  model.REC.V_m[1] = 0*mV # spike postsyn
    model.REC.V_m[0] = 0*mV
    model.run(abs(i))
    model.REC.V_m[0] = 0*mV
    model.run(abs(i))
   # model.REC.V_m[0] = 0*mV
    model.run(abs(i))
   # model.REC.V_m[0] = 0*mV
    model.run(abs(i))
    #model.REC.V_m[0] = 0*mV
    #model.run(abs(i))
    model.run(30*ms)
   #model.REC.V_m[1] = 0*mV
    #model.REC.V_m[0] = 0*mV
    #model.run(abs(i))
    #model.REC.V_m[0] = 0*mV''
else:
    model.REC.V_m[1] =0*mV  # spike presyn
    model.run(abs(i))       # time between 
    model.REC.V_m[0] = 0*mV # spike postsyn


time_left = time_after - (5*ms + abs(i))
model.run(time_left)
w_after = model.S_REC.w[0] # index at timestep of 100 ms 
model.run(model_run_length*ms)


print('spiking intervall', i, 'ms')
print('weight after 5 ms:', w_before)
print('weight after 100ms: ', w_after)
print('delta weight: ', (w_after - w_before))
model_run_length = model_run_length + 100 + 5 


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 3, 3]}, sharex=True)


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
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='red', marker='|', linestyle='None', markersize=14, markeredgewidth=4.5, label='Pre-synaptic spike (0)'),
    Line2D([0], [0], color='blue', marker='|', linestyle='None', markersize=14, markeredgewidth=4.5, label='Post-synaptic spike (1)')
]
ax1.legend(handles=legend_elements, fontsize=12)
ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

ax2.plot(weightmon.t/ms, weightmon.w[0]) #t/ms
ax2.legend(fontsize=10)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Weight')
#ax2.set_xticks(range(0, model_run_length+1, 10))
ax2.set_xticks(range(0, model_run_length+1, 50))
ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
ax2.axvline(x=100, color='black', linewidth=1.5, linestyle='--')
#ax2.axhline(x=100, color='black', linewidth=1.5, linestyle='--')


ax3.plot(biasmon.t/ms, biasmon.beta[0], color='red', label='Pre (0)')
ax3.plot(biasmon.t/ms, biasmon.beta[1], color='blue', label='Post (1)')


plt.tight_layout()
plt.show()