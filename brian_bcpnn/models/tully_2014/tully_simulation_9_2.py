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
#P_syn_values = [1.15*epsilon_n**2, 1.25*epsilon_n**2, 1.35*epsilon_n**2, 1.4*epsilon_n**2]
#P_syn_values = [1.9*epsilon_n**2, 2.1*epsilon_n**2, 2.3*epsilon_n**2, 2.5*epsilon_n**2]
#P_syn_values = [1.05*epsilon_n**2, 1.1*epsilon_n**2, 1.2*epsilon_n**2, 1.25*epsilon_n**2]
P_syn_values = [2*epsilon_n**2,3*epsilon_n**2,4*epsilon_n**2, 5*epsilon_n**2, 6*epsilon_n**2, 10*epsilon_n**2, 11*epsilon_n**2, 12*epsilon_n**2]  
#weight_traces = {}  # stores {P_syn: (t, w)} per run
#bias_traces = {}
#P_syn_values = [epsilon_n**2, epsilon_n**2, epsilon_n**2, epsilon_n**2, epsilon_n**2  ]#trying all same initialized weight


spiking_intervals = [201, 50, 20, 10, 5] # (ms), For 20, 50, 100, 200 Hz respectively 
nbr_of_intervals = len(spiking_intervals) 
all_results = {} # {P_syn: {interval: (w_start, w_end)}}

for interval in tqdm(spiking_intervals):
    weight_traces = {}  #resets for each run
    for P_syn in tqdm(P_syn_values):

        start_scope() 
        time_after=100*ms
        model = TullyNetwork()
        tully_namespace['tau_p'] = NEW_TAU_P
        tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)
        tully_namespace['f_max'] = 30*Hz
        tully_namespace['epsilon'] = epsilon_n
        tully_namespace['K'] = 0.1 # decrease to decrease plasticity

        eps = epsilon_n
        model.S_REC.set_states({
            'Z_i': eps, 'E_i': eps, 'P_i': 1.25*eps,
            'E_syn': eps**2, 'P_syn': P_syn  
        })

        model.REC.set_states({
            'Z_j': eps, 'E_j': eps, 'P_j': 1.25*eps
        })

        weightmon = model.add_synmon(variables=['w'], record=True)
        spikemon = model.add_spikemon()
        #biasmon = StateMonitor(model.REC, 'beta', record=True)
        biasmon = StateMonitor(source=model.REC, variables='beta', record=True)
        model.add_monitor(biasmon, biasmon.name)
       # w_start = weightmon.w[0][0] 

        # --- CONTROL: run same duration but no spikes ---
        model.run(5*ms)
        model.run(3 * interval * ms)  # same duration as the spike runs
        time_after_run = 100*ms 
        model.run(time_after_run)
        w_control = weightmon.w[0][-1]  # weight after same time, no spikes

            # CONTROOOOL
        start_scope()
        model = TullyNetwork()
        tully_namespace['tau_p'] = NEW_TAU_P
        tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)
        tully_namespace['f_max'] = 30*Hz
        tully_namespace['epsilon'] = epsilon_n
        tully_namespace['K'] = 0.1
        model.S_REC.set_states({
            'Z_i': eps, 'E_i': eps, 'P_i': eps,
            'E_syn': eps**2, 'P_syn': P_syn  
        })
        model.REC.set_states({
            'Z_j': eps, 'E_j': eps, 'P_j': eps
        })
        weightmon = model.add_synmon(variables=['w'], record=True)
        spikemon = model.add_spikemon()
        biasmon = StateMonitor(source=model.REC, variables='beta', record=True)
        model.add_monitor(biasmon, biasmon.name)

        model.run(5*ms)
        model.REC.V_m[0] = 0*mV 
        model.run(interval * ms)
        model.REC.V_m[0] = 0*mV
        model.run(interval * ms)
        model.REC.V_m[0] = 0*mV
        model.run(interval * ms)
        
        time_after_run = 100*ms 
        model.run(time_after_run)
        w_end = weightmon.w[0][-1]

        # normalize against control
        if P_syn not in all_results:
            all_results[P_syn] = {}
        all_results[P_syn][interval] = (w_control, w_end)



'''
        model.run(5*ms) # run 5 ms first
        w_0 = float(np.log(P_syn) - np.log(epsilon_n**2))
        print(f'w_0 analytical: {w_0:.4f}')
        print(f'w monitor at t=0: {weightmon.w[0][0]:.4f}')
        print(f'P_syn: {P_syn:.6e}, epsilon_n**2: {epsilon_n**2:.6e}')
         
        # three presynapic spikes at different time intervals -> different frequencies 

        model.REC.V_m[0] = 0*mV 
        model.run(interval * ms)
        model.REC.V_m[0] = 0*mV
        model.run(interval * ms)
        model.REC.V_m[0] = 0*mV
        model.run(interval * ms )
   
        # trying
        #model.run(interval * ms)
        #model.REC.V_m[0] = 0*mV

        time_after_run = 200*ms #?
        model.run(time_after_run)

        #w_before = weightmon.w[0][0]                                             
        #w_after_100 = weightmon.w[0][np.searchsorted(weightmon.t, 100*ms)] 
       # w_start = weightmon.w[0][0]  # baseline from monitor at t=0
        #w_end = weightmon.w[0][np.searchsorted(weightmon.t, 100*ms)]
                                               #(interval*3 + 100)*ms)]
        w_end = weightmon.w[0][-1]

        idx = np.searchsorted(weightmon.t, 100*ms)
        print(f'Index: {idx}, time at index: {weightmon.t[idx]/ms} ms, total length: {len(weightmon.t)}')
        w_0 = float(np.log(P_syn) - np.log((1.08*1.08*epsilon_n*epsilon_n)))  # OBS CHANGE PI PJ MANUALLY 
       # weight_traces[P_syn] = (w_0, w_before, w_after_100)

        if P_syn not in all_results:
            all_results[P_syn] = {}
        all_results[P_syn][interval] = (w_start, w_end)
'''


#all_delta_w = [dw for _, _, delta_w in all_results.values() for dw in delta_w]
#max_delta_w = np.max(np.abs(all_delta_w))

    
freq_list = []
j = 0


for P_syn, interval_results in all_results.items():
    sorted_items = sorted(interval_results.items())  # sort by interval
    w_starts = [v[0] for _, v in sorted_items]
    w_ends   = [v[1] for _, v in sorted_items]
    y = [end / start * 100 for start, end in zip(w_starts, w_ends)]
    freq_list = [int(1000 / interval) for interval, _ in sorted_items]
    plt.plot(freq_list, y, marker='o') #, label=f'w_0={w_0:.2e}'


font1 = {'family':'sans-serif','size':15}
font2 = {'family':'sans-serif','size':22}

#plt.axhline(y=0, color='grey', linewidth=1.5, linestyle='--')
plt.xlabel('Spiking frequency (Hz)', fontdict=font1)
#plt.ylabel('Δw/max(Δw)', fontdict=font1)
#plt.ylabel(r'$\frac{\Delta w}{max(\Delta w)}$', fontdict=font2)
plt.ylabel('Normalized weight (%)', fontdict=font1)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.legend()
plt.show()