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
from brian_bcpnn.models.tully_2014.tully_params import tully_equations
# from activation_patterns import activation_lists

NEW_TAU_P = 100*ms # 3 000 ms , 200 ms
dt = 0.01 * ms
defaultclock.dt = dt
epsilon_n = 0.00033 # epsilon = 1 /(f_max * tau_p) = 0.00033
# new_tau_e = 100*ms
# trying new tau_e tau_p values 

# --------- SINGLE SPIKE PAIRS ------------

i = 10*ms # spike timing interval
# i = 5*ms
# i = -5*ms
# i = 20*ms
# i = 30*ms
start_scope()
model = TullyNetwork()
model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], NEW_TAU_P, model.N_H, model.N_M)
model.namespace['tau_p'] = NEW_TAU_P # add in loop also
model.namespace['epsilon'] = epsilon_n
# model.namespace['tau_e'] = new_tau_e
weightmon = model.add_synmon(variables=['w'], record=True)
spikemon = model.add_spikemon()
# w_before = model.S_REC.w[0]
model.run(5*ms)
if i > 0:
    model.REC.V_m[0] = 0*mV # spike presyn
    model.run(abs(i))            # time between 
    model.REC.V_m[1] = 0*mV # spike postsyn
else:
    model.REC.V_m[1] =0*mV  # spike presyn
    model.run(abs(i))       # time between 
    model.REC.V_m[0] = 0*mV # spike postsyn

# w_after = model.S_REC.w[0]
model.run(NEW_TAU_P) # run 2xi ?x


'''
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
trains.compare_two_trains(ax1, spikemon, 0, 1, t_div=NEW_TAU_P)
# ax1.plot(spikemon.t, spikemon)
ax2.plot(weightmon.t/ms, weightmon.w[0])
# plot(spikemon.t/ms, spikemon)
plt.grid()
#plt.legend() 

# composite.plot_traces(0, 1, weightmon, model.S_REC, t_div=ms)
plt.show()
'''

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# Raster plot - scatter of (time, neuron index)
ax1.scatter(spikemon.t/ms, spikemon.i, marker='|', s=200, c='black')
ax1.set_ylabel('Neuron index')
ax1.set_yticks([0, 1])
ax1.set_yticklabels(['Pre (0)', 'Post (1)'])
ax1.set_title('Spike raster')
ax1.scatter(spikemon.t[spikemon.i == 0]/ms, spikemon.i[spikemon.i == 0], 
            marker='|', s=200, c='red', label='Pre (0)')
ax1.scatter(spikemon.t[spikemon.i == 1]/ms, spikemon.i[spikemon.i == 1], 
            marker='|', s=200, c='blue', label='Post (1)')
ax1.legend()

ax2.plot(weightmon.t/ms, weightmon.w[0])
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Weight')


plt.tight_layout()
plt.show()
