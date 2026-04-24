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
dt = 0.1 * ms
defaultclock.dt = dt
t_total = 10*NEW_TAU_P # 10 * tau_p = 30 000 ms = 30 s
t_stim = t_total/50 


'''
# STIMULI
neuron_1_coords = ColumnCoords(0, 0) #presynaptic neuron
neuron_2_coords = ColumnCoords(1, 0) #postsynaptic neuron

model = TullyNetwork() #Everything defined. Subclass of CorticalNetwork.
namespace = model.namespace # using defined attribute namespace from the TullyNetwork. 
# namespace used in Brian as dictionary of defined parameters and values. 

figure_2_stims = [
    StimProtocol(neuron_1_coords, StimTime(t_start=400*ms, t_end=420*ms)),   #pre-syn neuron 0-100ms
    StimProtocol(neuron_2_coords, StimTime(t_start=420*ms, t_end=440*ms)), #post-syn neuron 100-200ms
]


# do I really want to stimulate neuron 2 or just activate it? but how? 
# Read more on StimTime and STimPRotocol

timed_array = stils.stim_times_to_timed_array(figure_2_stims, t_total, 2, 1)
model.namespace['stim_ta'] = timed_array 


# model.namespace['stim_ta'] = stils.stim_times_to_timed_array(figure_2_stims, t_total, model.N_H, model.N_M)


# Difference between stimulus and spike.. How should I give it a spike at specific times ? 
# I try giving it different much stimulation and seeing how much and when it spikes and then adapt?
# Only want one presynaptic spike followed by one postsynaptic spike. 

# monitors

spikemon = model.add_spikemon()
weightmon = model.add_synmon(variables=['w'], record=True)
#where does it get the neuron group from?

model.run(t_total)

# running for interspiking times
# setting voltages to zero
# letting it run for the whole time
# calculating weights

# will need to plot weight after, but for now only want to plot spikemon

# from tully sim 1: composite.plot_traces(0, 1, spikemon, tracemon, syn_tracemon, model.S_REC, t_div=ms)

#spikemon = SpikeMonitor(G)

#run(50*ms)

print('Spike times: %s' % spikemon.t[:])

fig, (ax1) = plt.subplots(1, 1, sharex=True)
# måste ändra här så det mäts från rätt stimulering och ej förra
# is the spikemon same as before the problem?
# I think the spike trains are the problem ..
# normally spikemonitor spikemon = SpikeMonitor(G) takes group to record from (G here)
# how is this defined in model? aka  model = TullyNetwork() (subclass of Cortical Network)

trains.compare_two_trains(ax1, spikemon, 0, 1, t_div=NEW_TAU_P)
plt.show()
# do not actually need to use compare_two_trains, can also just plot both individually..
# look at documentation. 
# when gotten spike monitor to work, then start measuring weight and also do so it only spikes once
# if this is possible..

# how long does the simulation run for ?

#------------------------------------------------------------------

# 400 - 430 ms
# Total simulation time = 5.28 seconds.
# Spike times: [0.4165 0.4249] s

# 400 - 420 ms
# Total simulation time = 7.29 seconds. - ?? and why increasing ??
# Spike times: [0.4069] s  = 406.9 ms 

# now with both neurons:
# 400-420 ms & 420-440 ms
# Total simulation time = 5.29 seconds.
# Spike times: [0.4067 0.4127 0.4266 0.4349] s

# another run
# Total simulation time = 5.35 seconds.
# spike times: [0.4126 0.4318] s

'''



# timed_array = TimedArray([0, 0] * mV, dt=0.1*ms)
# look up t_total (and other arguments) and what Ferdinand wrote about 


t_int = 10*ms
interval_list = range(-50, 50+1) # think +1 should be the way to go?
delta_w_list = []
for i in tqdm(interval_list): # loading bar 
    start_scope()
    model = TullyNetwork()
    model.namespace['stim_ta'] = stils.stim_times_to_timed_array([], t_total, model.N_H, model.N_M) # creating empty TimedArray to run
   
    w_before = model.S_REC.w[0]
    model.run(5*ms)

    if i > 0: # aka positive
        # pre - before - post spiking 
        model.REC.V_m[0] = 0*mV # pre first. enough to spike, over -55mV enough (?)
        model.run(i*ms) # t_int before
        # post - before - pre
        model.REC.V_m[1] = 0*mV # -|| - 
        model.run(3*ms) # or something, enough after the spike!

    else: # aka negative
        # post - before - pre spiking
        model.REC.V_m[1] = 0*mV # post first
        model.run(abs(i)*ms) # t_int # abs right I think?
        model.REC.V_m[0] = 0*mV # pre second
        model.run(3*ms)
    
  #  w_after = model.S_REC.w[0] # weight of the first synapse, aka the one connecting neuron 0 and 1 
    w_after = np.max(model.S_REC.w[0]) # does this work, highest the weight becomes ?
    delta_w_list.append(w_after - w_before) # why does this never turn negative, even for large delta values? 
    yaxis_delta_w = delta_w_list / np.max(delta_w_list) # should do this in or outside of the loop?

plt.figure(figsize=(8, 5))
plt.plot(interval_list, yaxis_delta_w, 'o-', color='steelblue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('Time interval Δt (ms)\npost→pre (negative) / pre→post (negative)')
plt.ylabel('Δw (weight change)')
plt.title('STDP curve')
#plt.tight_layout()
plt.show()
    
    # weightmon = model.add_synmon(variables=['w'], record=True)

    # FIGURE OUT NOW HOW TO PLOT THE WEIGHT MONITOR

   # w_array[i,:]
    # spikemon = model.add_spikemon()
    
# why am I getting such weird simulation times and multiple?

# --- IN TULLY SIM 2 WE HAVE:
# n_iterations = 10
# w_array = np.zeros(shape=(n_iterations, int(t_total/defaultclock.dt)))
# in the for - loop:
#    w_array[i,:] = weightmon.w[0] # but I do not think I want to use the i in the same way here?

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