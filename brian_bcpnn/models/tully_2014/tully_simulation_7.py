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

#i = 10*ms # spike timing interval
i = 5*ms

# ----- SCATTER PLOT OF INTIAL WEIGHT VS WEIGHT AFTER 100 MS:--------
# ------- MULTIPLE SCATTER PLOTS!!

start_scope()

#P_syn_values = [30*epsilon_n**2, 40*epsilon_n**2, 50*epsilon_n**2, 60*epsilon_n**2, 70*epsilon_n**2]  
#P_syn_values = [15*epsilon_n**2, 20*epsilon_n**2, 25*epsilon_n**2, 30*epsilon_n**2, 35*epsilon_n**2, 40*epsilon_n**2]
P_syn_values = [1.005*epsilon_n**2, 1.1*epsilon_n**2, 1.35*epsilon_n**2, 1.5*epsilon_n**2, 1.8*epsilon_n**2,2*epsilon_n**2, 2.5*epsilon_n**2, 3*epsilon_n**2, 3.5*epsilon_n**2, 4*epsilon_n**2, 4.5*epsilon_n**2, 5*epsilon_n**2, 5.5*epsilon_n**2, 6*epsilon_n**2, 6.5*epsilon_n**2, 7*epsilon_n**2, 7.5*epsilon_n**2] #, 8*epsilon_n**2, 8.5*epsilon_n**2]
#weight_traces = {}  # stores {P_syn: (t, w)} per run
#bias_traces = {}

spike_counts = [1,2,3,4,5,6,7,8]
all_results = {} # {n_spikes: (w_starts, w_ends)}

for n_spikes in tqdm(spike_counts):
    weight_traces = {}  #resets for each run

    for P_syn in P_syn_values:

        start_scope() 
        time_after=100*ms
        tully_namespace['epsilon'] = epsilon_n
        tully_namespace['K'] = 1 # decrease to decrease plasticity
        model = TullyNetwork()
    # tully_namespace['epsilon'] = epsilon_n
        tully_namespace['tau_p'] = NEW_TAU_P
        tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)

    
        eps = epsilon_n
        model.S_REC.set_states({
            'Z_i': eps, 'E_i': eps, 'P_i': eps,
            'E_syn': eps**2, 'P_syn': P_syn  
        })

        model.REC.set_states({
            'Z_j': eps, 'E_j': eps, 'P_j': eps
        })

        weightmon = model.add_synmon(variables=['w'], record=True)
        spikemon = model.add_spikemon()
        #biasmon = StateMonitor(model.REC, 'beta', record=True)
        biasmon = StateMonitor(source=model.REC, variables='beta', record=True)
        model.add_monitor(biasmon, biasmon.name)

        model.run(5*ms)

        for _ in range(n_spikes):
            model.REC.V_m[0] = 0*mV
            model.run(abs(i))

        time_after_run = (100*ms) - (5*ms + abs(i))
        model.run(time_after_run)
        model.run(model_run_length * ms)
        # -----------------------------------------

        w_before = weightmon.w[0][0]                                             
        w_after_100 = weightmon.w[0][np.searchsorted(weightmon.t, 100*ms)] 
        idx = np.searchsorted(weightmon.t, 100*ms)
        print(f'Index: {idx}, time at index: {weightmon.t[idx]/ms} ms, total length: {len(weightmon.t)}')
        w_0 = float(np.log(P_syn) - np.log((eps*eps)))  # OBS CHANGE PI PJ MANUALLY 
        weight_traces[P_syn] = (w_0, w_before, w_after_100)

    w_starts = [v[1] for v in weight_traces.values()] # initial weight
    w_ends   = [v[2] for v in weight_traces.values()] 
    delta_w = [end - start for start, end in zip(w_starts, w_ends)]
    all_results[n_spikes] = (w_starts, w_ends, delta_w) # store per spike count

#for n_spikes, (w_starts, w_ends, delta_w) in all_results.items():


    
max_delta_w = np.max(delta_w)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

for n_spikes, (w_starts, w_ends, delta_w) in all_results.items():
    y_axis_ax2 = delta_w/max_delta_w
    ax1.plot(w_starts, w_ends, marker='o', label=f'{n_spikes} spike(s)')
    ax2.plot(w_starts, y_axis_ax2, marker='o', label=f'{n_spikes} spike(s)')


# diagonal reference line across all data
all_w = [w for w_starts, _, _ in all_results.values() for w in w_starts]
ax1.plot([min(all_w), max(all_w)], [min(all_w), max(all_w)], 
         'k--', linewidth=1, label='y = x (no change)')
ax1.axhline(y=0, color='grey', linewidth=1.5, linestyle='--')
ax1.set_xlabel('Initial weight (w₀)')
ax1.set_ylabel('w after 100 ms')
ax1.legend(fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)

# diagonal reference line across all data
all_w = [w for w_starts, _, _ in all_results.values() for w in w_starts]
#ax2.plot([min(all_w), max(all_w)], [min(all_w), max(all_w)], 
#         'k--', linewidth=1, label='y = x (no change)')
ax2.axhline(y=0, color='grey', linewidth=1.5, linestyle='--')
ax2.set_xlabel('Initial weight (w₀)')
ax2.set_ylabel('Δw/max(Δw)')
ax2.legend(fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

