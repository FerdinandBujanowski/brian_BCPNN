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

#time_after = 100*ms
time_after = 100*ms
NEW_TAU_P = 1000*ms # As in Tully.  
#NEW_TAU_P = 1000*ms 
dt = 0.01 * ms
defaultclock.dt = dt
epsilon_n = 0.0033 # epsilon = f_min/f_max, a baseline firing rate #0.033
# before epsilon = 1 /(f_max * tau_p) = 0.0033
model = TullyNetwork
tully_namespace['epsilon'] = epsilon_n
#tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M) # creating empty TimedArray to run with
#tully_namespace['tau_p'] = NEW_TAU_P 
tully_namespace['tau_z_i'] = 5*ms
tully_namespace['tau_z_j'] = 5*ms

# --------------------- STDP PLOT ------------------------------


interval_list = range(-50, 51) # +1 ?
#interval_list = range(-10, 10)
#delta_w_list = []
w_after_list = []
#j = -1

tau_p_vals = [1000*ms, 2000*ms]
colors = ['steelblue', 'tomato']
# labels # later

plt.figure(figsize=(8,5))

for tau_p_val, color in zip(tau_p_vals, colors): #, labels):

    delta_w_list = []
    j = -1

    for i in tqdm(interval_list): # tqdm : loading bar #in [40]
        start_scope()
        time_after = 100*ms
        model = TullyNetwork()                                          # changed from NEW_TAU_P to time_total
        model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M) # creating empty TimedArray to run with
        model.namespace['tau_p'] = tau_p_vals
        weightmon = model.add_synmon(variables=['w'], record=True)
        model.run(5*ms)
        w_before = model.S_REC.w[0] # S_REC is the synapse between i and j, so this is the synaptic strength between neuron 0 and 1 aka the weight. Weight of the first synapse. 
    #  print('weight before: ', w_before)

        if i > 0: # aka positive
            # pre - before - post spiking 
            model.REC.V_m[0] = 0*mV # pre first. enough to spike, over -55mV enough (?)
            model.run(abs(i)*ms) # 
            model.REC.V_m[1] = 0*mV # post second.

        else: # aka negative
            # post - before - pre spiking
            model.REC.V_m[1] = 0*mV # post first
            model.run(abs(i)*ms) # abs value
            model.REC.V_m[0] = 0*mV # pre second


        time_after = time_after - (5 + abs(i))*ms
        model.run(time_after) # or NEW_TAU_P - abs(i) - 5 ms ?
        w_after = model.S_REC.w[0] # index at timestep of 100 ms 

        delta_w_list.append(w_after - w_before)
        j += 1

    max_delta = np.max(delta_w_list)
    axis_delta_w = delta_w_list / max_delta

    plt.plot(interval_list, axis_delta_w, 'o-', color=color)

#plt.figure(figsize=(8, 5))
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time interval Δt (ms), between first and second spike.') # \npost→pre (negative) / pre→post (positive)')
plt.ylabel('Δw_{ij} / max(Δ_w_{ij}')
#' Δw = w_after - w_before ')
plt.title('STDP curve')
plt.xticks(range(-50, 51, 5))
plt.yticks(np.arange(-5, 5, 0.5))
plt.grid(True, linestyle='--', alpha=0.5)
plt.axhline(y=0, color='red', linewidth=1, linestyle='--')
plt.axvline(x=0, color='red', linewidth=1, linestyle='--')
plt.legend()

plt.show()
