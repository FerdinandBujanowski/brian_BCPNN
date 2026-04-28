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

NEW_TAU_P = 3*second # 3 000 ms 
dt = 0.01 * ms
defaultclock.dt = dt
t_total = 10*NEW_TAU_P # 10 * tau_p = 30 000 ms = 30 s
t_stim = t_total/50 


# look up t_total (and other arguments) and what Ferdinand wrote about 

i = 40*ms # spike timing interval
start_scope()
model = TullyNetwork()
model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], t_total, model.N_H, model.N_M)
w_before = model.S_REC.w[0]
model.run(5*ms)
model.REC.V_m[0] = 0*mV # spike
model.run(i)            # time between 
model.REC.V_m[1] = 0*mV # spike
w_after = model.S_REC.w[0]
weightmon = model.add_synmon(variables=['w'], record=True)
# spikemon = model.add_spikemon()

model.run(t_total)


#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#trains.compare_two_trains(ax1, spikemon, 0, 1, t_div=NEW_TAU_P)
# ax1.plot(spikemon.t, spikemon)
plot(weightmon.t, weightmon.w[0])
# composite.plot_traces(0, 1, weightmon, model.S_REC, t_div=ms)
plt.show()




'''
t_int = 10*ms
interval_list = range(-50, 50) # think +1 should be the way to go?
delta_w_list = []
for i in tqdm(interval_list): # tqdm -> loading bar 
    start_scope()
    model = TullyNetwork()
    model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], t_total, model.N_H, model.N_M) # creating empty TimedArray to run with
    w_before = model.S_REC.w[0] # S_REC is the synapse between i and j, so this is the synaptic strength aka the weight.
    model.run(5*ms)

    if i > 0: # aka positive
        # pre - before - post spiking 
        model.REC.V_m[0] = 0*mV # pre first. enough to spike, over -55mV enough (?)
        model.run(i*ms) # t_int before
        # post - before - pre
        model.REC.V_m[1] = 0*mV # post second.
        model.run(3*ms) # or something, enough after the spike!

    else: # aka negative
        # post - before - pre spiking
        model.REC.V_m[1] = 0*mV # post first
        model.run(abs(i)*ms) # t_int # abs right I think?
        model.REC.V_m[0] = 0*mV # pre second
        model.run(3*ms) # for how long after? 
    
  #  w_after = model.S_REC.w[0] # weight of the first synapse, aka the one connecting neuron 0 and 1 
    w_after = np.max(model.S_REC.w[0]) # does this work, highest the weight becomes ?
    delta_w_list.append(w_after - w_before) # why does this never turn negative, even for large delta values? 
    yaxis_delta_w = delta_w_list / np.max(delta_w_list) # should do this in or outside of the loop?

plt.figure(figsize=(8, 5))
plt.plot(interval_list, yaxis_delta_w, 'o-', color='steelblue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time interval Δt (ms)\npost→pre (negative) / pre→post (negative)')
plt.ylabel('Δw / max Δw ')
plt.title('STDP curve')
#plt.tight_layout()
plt.show()

'''


# decide how long to run it for etc

    # then need to add weight monitor 
    # need to measure maximal weight? with np.max
    # if interval negative, post-before-pre
    # if interval positive, pre-before-post

    # need to initialize the weight as 1, the % diff will be wrong otherwise I think. 
    # w = 1 = log(Psyn (= Pij) / Pi x Pj) <=> Pi, Pj = eps ? and Pij = 10x eps^2 
    # fix interval list haha and also increments. Increments of 1 ms ? 

    # np.max for max weight etc......



'''        self.S_REC = Synapses(
            self.REC, self.REC, model=eqs['bcpnn_syn_model'], on_pre=eqs['bcpnn_syn_on_pre'], method='euler', delay=self.namespace['t_delay']

            self.S_REC.connect(i=source_rec, j=target_rec)
        )'''