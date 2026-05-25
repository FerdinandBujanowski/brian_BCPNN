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
model_run_length = 500 # this is? 

#i = 10*ms # spike timing interval
i = 10*ms

# --------  SCATTERPLOT OF DELTA_W VS SPIKE FREQUENCY

start_scope()

#P_syn_values = [30*epsilon_n**2, 40*epsilon_n**2, 50*epsilon_n**2, 60*epsilon_n**2, 70*epsilon_n**2]  
#P_syn_values = [15*epsilon_n**2, 20*epsilon_n**2, 25*epsilon_n**2, 30*epsilon_n**2, 35*epsilon_n**2, 40*epsilon_n**2]
#P_syn_values = [1.005*epsilon_n**2, 1.1*epsilon_n**2, 1.35*epsilon_n**2, 1.5*epsilon_n**2, 1.8*epsilon_n**2,2*epsilon_n**2, 2.5*epsilon_n**2, 3*epsilon_n**2, 3.5*epsilon_n**2, 4*epsilon_n**2, 4.5*epsilon_n**2, 5*epsilon_n**2, 5.5*epsilon_n**2, 6*epsilon_n**2, 6.5*epsilon_n**2, 7*epsilon_n**2, 7.5*epsilon_n**2] #, 8*epsilon_n**2, 8.5*epsilon_n**2]
#P_syn_values = [1.005*epsilon_n**2, 1.1*epsilon_n**2, 1.2*epsilon_n**2, 1.25*epsilon_n**2, 1.30*epsilon_n**2, 1.35*epsilon_n**2, 1.5*epsilon_n**2, 1.8*epsilon_n**2, 2*epsilon_n**2, 2.2*epsilon_n**2, 2.5*epsilon_n**2, 3*epsilon_n**2]
P_syn_values = [1.005*epsilon_n**2, 1.1*epsilon_n**2, 1.2*epsilon_n**2, 1.25*epsilon_n**2]
#weight_traces = {}  # stores {P_syn: (t, w)} per run
#bias_traces = {}
#P_syn_values = [epsilon_n**2, epsilon_n**2, epsilon_n**2, epsilon_n**2, epsilon_n**2  ]#trying all same initialized weight


spiking_intervals = [50, 20, 10, 5] # (ms), For 20, 50, 100, 200 Hz respectively 
nbr_of_intervals = len(spiking_intervals) 
all_results = {} # {n_spikes: (w_starts, w_ends)}

for interval in tqdm(spiking_intervals):
    weight_traces = {}  #resets for each run
    for P_syn in tqdm(P_syn_values):

        start_scope() 
        time_after=100*ms
        model = TullyNetwork()
        tully_namespace['tau_p'] = NEW_TAU_P
        tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)
        tully_namespace['f_max'] = 100*Hz
        tully_namespace['epsilon'] = epsilon_n
        tully_namespace['K'] = 0.1 # decrease to decrease plasticity

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

        model.run(5*ms) # run 5 ms first
         
        # three presynapic spikes at different time intervals -> different frequencies 

        model.REC.V_m[0] = 0*mV 
        model.run(interval * ms)
        model.REC.V_m[0] = 0*mV
        model.run(interval * ms)
        model.REC.V_m[0] = 0*mV
        model.run(interval * ms )

        time_after_run = 300*ms #?
        model.run(time_after_run)

        w_before = weightmon.w[0][0]                                             
        w_after_100 = weightmon.w[0][np.searchsorted(weightmon.t, 100*ms)] 
        idx = np.searchsorted(weightmon.t, 100*ms)
        print(f'Index: {idx}, time at index: {weightmon.t[idx]/ms} ms, total length: {len(weightmon.t)}')
        w_0 = float(np.log(P_syn) - np.log((eps*eps)))  # OBS CHANGE PI PJ MANUALLY 
        weight_traces[P_syn] = (w_0, w_before, w_after_100)

    w_starts = [v[1] for v in weight_traces.values()] # initial weight
    w_ends   = [v[2] for v in weight_traces.values()] 
    delta_w = [end - start for start, end in zip(w_starts, w_ends)]
    j=0
    for j in range(nbr_of_intervals):
        all_results[j] = (w_starts, w_ends, delta_w) # store per spike count


all_delta_w = [dw for _, _, delta_w in all_results.values() for dw in delta_w]
max_delta_w = np.max(np.abs(all_delta_w))
    
freq_list = []
j = 0
for j, (w_starts, w_ends, delta_w) in all_results.items():
    spiking_intervals
    hz_value = int((3 / (3*spiking_intervals[j])) * 1000) # total time interval of spikes now
    freq_list.append(hz_value)


for j, (w_starts, w_ends, delta_w) in all_results.items():
    y_axis_ax2 = delta_w/max_delta_w # normalized
    #hz_value = int((n_spikes / 60) * 1000) # total time interval of spikes now
    plt.plot(freq_list, y_axis_ax2, marker='o') #,freq_list[n_spikes-1]


font1 = {'family':'sans-serif','size':15}

plt.axhline(y=0, color='grey', linewidth=1.5, linestyle='--')
plt.xlabel('Spiking frequency (Hz)', fontdict=font1)
plt.ylabel('Δw/max(Δw)', fontdict=font1)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()