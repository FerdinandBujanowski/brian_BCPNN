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

## --- MULTIPLE WEIGHT TRACES (W VS TIME) OVER SEVERAL SPIKES

start_scope()

#P_syn_values = [1.15*epsilon_n**2,1.5*epsilon_n**2,2*epsilon_n**2, 3*epsilon_n**2, 4*epsilon_n**2, 5*epsilon_n**2, 7*epsilon_n**2, 10*epsilon_n**2]  
#P_syn_values = [2*epsilon_n**2, 2.71828*epsilon_n**2, 3*epsilon_n**2]
P_syn_values = [np.exp(0.3)*epsilon_n**2,np.exp(0.4)*epsilon_n**2, np.exp(0.5)*epsilon_n**2, np.exp(0.6)*epsilon_n**2, np.exp(0.7)*epsilon_n**2, np.exp(0.8)*epsilon_n**2, np.exp(0.9)*epsilon_n**2, np.exp(1)*epsilon_n**2]
#P_syn_values = [1.005*epsilon_n**2, 1.1*epsilon_n**2, 1.2*epsilon_n**2, 1.25*epsilon_n**2]
#P_syn_values = [20*epsilon_n**2, 30*epsilon_n**2, 40*epsilon_n**2, 50*epsilon_n**2]
weight_traces = {}  # stores {P_syn: (t, w)} per run
bias_traces = {}

#i = 5*ms
i = 5*ms

for P_syn in tqdm(P_syn_values):

    start_scope() 
    time_after=100*ms
    tully_namespace['epsilon'] = epsilon_n
    model = TullyNetwork()
   # tully_namespace['epsilon'] = epsilon_n
    tully_namespace['tau_p'] = NEW_TAU_P
    tully_namespace['stim_ta'] = stils.stim_times_to_timed_array([], time_after, model.N_H, model.N_M)
    tully_namespace['K'] = 0.065    # 0.035
    tully_namespace['f_max'] = 50 * Hz # trying 
   # tully_namespace['f_min'] = 49 * Hz


    eps = epsilon_n
  #  model.S_REC.Z_i = 1.0 + eps # full firing, trying 
    model.REC.set_states({
        'Z_j': eps, 'E_j': eps, 'P_j': eps
    })

    model.S_REC.set_states({
        'Z_i': eps, 'E_i': eps, 'P_i': eps,
        'E_syn': eps**2, 'P_syn': P_syn  
    })


    weightmon = model.add_synmon(variables=['w'], record=True)
    spikemon = model.add_spikemon()
    #biasmon = StateMonitor(model.REC, 'beta', record=True)
    biasmon = StateMonitor(source=model.REC, variables='beta', record=True)
    model.add_monitor(biasmon, biasmon.name)

    # 
    model.run(5*ms)

    #spiking_intervals = [50, 20, 10, 5] # (ms), For 20, 50, 100, 200 Hz respectively 
    i = 10*ms # 100 Hz frekvens

    if i > 0:
        model.REC.V_m[0] = 0*mV
        model.run(abs(i))
        model.REC.V_m[0] = 0*mV
        model.run(abs(i))
        model.REC.V_m[0] = 0*mV
        model.run(abs(i))
    #    model.REC.V_m[0] = 0*mV
   #     model.run(abs(i))
    #    model.REC.V_m[0] = 0*mV
    #    model.run(abs(i))
    #    model.REC.V_m[0] = 0*mV
    #    model.run(abs(i))
    #    model.REC.V_m[0] = 0*mV
    #    model.run(abs(i))



       # model.REC.V_m[1] = 0*mV # THREE PRE FOLLOWED BY ONE POST !
       # model.run(abs(i))
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

    w_0 = float(np.log(P_syn) - np.log((epsilon_n) * (epsilon_n)))  # OBS HERE CHANGE PI PJ MANUALLY 
  #  actual_P_syn = model.S_REC.P_syn[0]
  #  actual_P_i   = model.S_REC.P_i[0]
  #  actual_P_j   = model.REC.P_j[1]   # post-synaptic neuron
  #  w_0 = np.log(actual_P_syn) - np.log(actual_P_i * actual_P_j)   
    #beta_0 = np.ln(P_i)
   # weight_traces[P_syn] = (weightmon.t/ms, weightmon.w[0].copy(), w_0)
   # bias_traces[P_syn] = (biasmon.t/ms, biasmon.beta[0].copy(), beta_0)
    weight_traces[P_syn] = (
    weightmon.t/ms,
    weightmon.w[0].copy(),
    biasmon.beta[0].copy(),  # pre-synaptic bias
    biasmon.beta[1].copy(),  # post-synaptic bias
    w_0,
  #  w_before,
  #  w_after_100
)

for P_syn, (t, w, beta0, beta1, w_0) in weight_traces.items():
    print(f'w_0={w_0:.2f}, beta0 range: {beta0.min():.4f} to {beta0.max():.4f}')


fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 7]}, sharex=True)



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


for P_syn, (t, w, beta0, beta1, w_0) in weight_traces.items():
    label = f'w₀ = {w_0:.2f}'
    ax2.plot(t, w, label=label)
  #  ax3.plot(t, beta0, linestyle='-',  label=f'Pre  {label}')
  #  ax3.plot(t, beta1, linestyle='--', label=f'Post {label}')


ax2.legend(fontsize=10)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Weight')
#ax2.set_xticks(range(0, model_run_length+1, 10))
ax2.set_xticks(range(0, model_run_length+1, 50))
ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
ax2.axvline(x=100, color='black', linewidth=1.5, linestyle='--')
#ax2.axhline(x=100, color='black', linewidth=1.5, linestyle='--')


# PLOTTING BIAS PLOT
#ax3.set_xlabel('Time (ms)')
#ax3.set_ylabel('Beta (bias)')
#ax3.set_xticks(range(0, model_run_length+1, 50))
#ax3.axvline(x=100, color='black', linewidth=1.5, linestyle='--')
#ax3.grid(True, axis='x', linestyle='--', alpha=0.5)
#ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
#ax3.legend(fontsize=12)

plt.tight_layout()
plt.show()