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

NEW_TAU_P = 10000*ms # As in Tully.  3 000 ms
dt = 0.01 * ms
defaultclock.dt = dt
epsilon_n = 0.033 # epsilon = f_min/f_max, a baseline firing rate
# epsilon = 1 /(f_max * tau_p) = 0.0033


# --------------------- STDP PLOT ------------------------------

time_after = 100*ms
interval_list = range(-50, 50) # think +1 should be the way to go?
delta_w_list = []
for i in tqdm(interval_list): # tqdm : loading bar 
    start_scope()
    model = TullyNetwork()
    model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], NEW_TAU_P, model.N_H, model.N_M) # creating empty TimedArray to run with
    model.namespace['tau_p'] = NEW_TAU_P 

    model.namespace['epsilon'] = epsilon_n # adjusted to tau_p
    # I have not changed in networks to 'P_syn': 1.6487*eps**2 and will try with this.

    weightmon = model.add_synmon(variables=['w'], record=True)
    model.run(5*ms)
    w_before = model.S_REC.w[0] # S_REC is the synapse between i and j, so this is the synaptic strength aka the weight. Weight of the first synapse. 

    if i > 0: # aka positive
        # pre - before - post spiking 
        model.REC.V_m[0] = 0*mV # pre first. enough to spike, over -55mV enough (?)
        model.run(i*ms) # 
        model.REC.V_m[1] = 0*mV # post second.

    else: # aka negative
        # post - before - pre spiking
        model.REC.V_m[1] = 0*mV # post first
        model.run(abs(i)*ms) # abs value
        model.REC.V_m[0] = 0*mV # pre second

    model.run(100*ms) # or NEW_TAU_P - abs(i) - 5 ms ?
    w_after = model.S_REC.w[0] # index at timestep of 100 ms 

# ----------------
 #   correct way of getting the maximum weight (maximum in abs value)
 #   index_w_after = np.argmax(abs(weightmon.w[0])) #argmax returns indice of max value 
 #  w_after = weightmon.w[0][index_w_after]

   # w_after = model.S_REC.w[0] # weight of the first synapse, aka the one connecting neuron 0 and 1 
    delta_w_list.append(w_after - w_before) # why does this never turn negative, even for large delta values? 


# BELOW: trying to make the divided by max delta w work: ...

#index_max_delta = np.argmax(np.abs(delta_w_list)) # indice for where abs value maximal (of list of delta values)
#max_delta = delta_w_list[index_max_delta] # retrieving this maximal delta
max_delta = np.max(delta_w_list)
axis_delta_w = delta_w_list / max_delta
# axis_delta_w = delta_w_list / np.max(abs(delta_w_list)) # outside of loop



plt.figure(figsize=(8, 5))
plt.plot(interval_list, axis_delta_w, 'o-', color='steelblue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time interval Δt (ms)\npost→pre (negative) / pre→post (negative)')
plt.ylabel('Δw / max Δw ')
plt.title('STDP curve')
#plt.tight_layout()
plt.show()


'''

# decide how long to run it for etc

    # if interval negative, post-before-pre
    # if interval positive, pre-before-post

            # NOTES OF S_REC

     self.S_REC = Synapses(
            self.REC, self.REC, model=eqs['bcpnn_syn_model'], on_pre=eqs['bcpnn_syn_on_pre'], method='euler', delay=self.namespace['t_delay']

            self.S_REC.connect(i=source_rec, j=target_rec)
        )'''