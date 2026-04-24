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
from activation_patterns import activation_lists

NEW_TAU_P = 3*second # 3 000 ms 
dt = 0.1 * ms
defaultclock.dt = dt
t_total = 10*NEW_TAU_P # 10 * tau_p 
t_stim = t_total/50
        
n_iterations = 10 #can I try changing this?
t_array = None
w_array = np.zeros(shape=(n_iterations, int(t_total/defaultclock.dt)))
#w1_array = np.zeros(shape=(n_iterations, int(t_total/defaultclock.dt)))
beta_array = np.zeros(shape=(n_iterations, int(t_total/defaultclock.dt)))
#print(w_array.shape)

# STIMULI
neuron_1_coords = ColumnCoords(0, 0) #presynaptic neuron
neuron_2_coords = ColumnCoords(1, 0) #postsynaptic neuron


# enumerates each integer in stimulus strings (both at same time w. zip).
# Then, if = 1, does something.. does what?
def get_stim_list(stimulus_list_1, stimulus_list_2):
    figure_4_stims = []
    for i, (c1, c2) in enumerate(zip(stimulus_list_1, stimulus_list_2)):
        if c1 == 1:
            figure_4_stims.append(
                StimProtocol(neuron_1_coords, StimTime(t_start=i*t_stim, t_end=(i+1)*t_stim))
            )
        if c2 == 1:
            figure_4_stims.append(
                StimProtocol(neuron_2_coords, StimTime(t_start=i*t_stim, t_end=(i+1)*t_stim))
            )
    return figure_4_stims

spikemon = None
# REPEATED SIMULATION
all_spike_trains = [] #TRY!
for i in tqdm(range(n_iterations)):
    model = TullyNetwork(verbose=False)
    model.namespace['tau_p'] = NEW_TAU_P #global overwriting
    S_pre, S_post = activation_lists(n_iterations)
    figure_4_stims = get_stim_list(S_pre, S_post)
    timed_array = stils.stim_times_to_timed_array(figure_4_stims, t_total, 2, 1)
    model.namespace['stim_ta'] = timed_array 

    # MONITORS
    weightmon = model.add_synmon(variables=['w'], record=True)
    weightmon2 = model.add_synmon(variables=['w'], record=True) #change to =1 if only want postsynaptic
    # model.add_monitor / statemon ?
    spikemon = model.add_spikemon()
    biasmon = StateMonitor(source=model.REC, variables = ['beta'], record=1) #record = 1 does so only the 2nd (=postsynaptic) neuron is registered?
    # TODO add statemon for bias
    model.add_monitor(biasmon, biasmon.name) #model.add_statemon(variables, record) in Ferdinand's pre-built function 
    model.run(t_total)

    w_array[i,:] = weightmon.w[0]
    all_spike_trains.append(spikemon.spike_trains()) #TRY!
    if t_array is None:
        t_array = weightmon.t/model.namespace['tau_p'] #x-axis: time divided by tau_p

    beta_array[i,:] = biasmon.beta[0]
    if beta_array is None:
        beta_array = biasmon.t/model.namespace['tau_p'] # x-axis: time divided by tau_p
    

w_mean = np.mean(w_array, axis=0)
w_std = np.std(w_array, axis=0)
n_std = 1.96

#w1_mean = np.mean(w1_array, axis=0)

beta_mean = np.mean(beta_array, axis=0)
beta_std = np.std(beta_array, axis=0)

# ax1 is spikemonitor 
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
trains.compare_two_trains(ax1, spikemon, 0, 1, t_div=NEW_TAU_P)

ax2.plot(t_array, w_mean, color='c', label='mean')
ax2.fill_between(t_array, w_mean-n_std*w_std, w_mean+n_std*w_std, color='c', alpha=0.3, label='95%')

ax3.plot(t_array, beta_mean, color='m', label='mean')
ax3.fill_between(t_array, beta_mean-n_std*beta_std, beta_mean+n_std*beta_std, color='m', alpha=0.3, label='95')

# do for ax4 delta weight change. so w_2 - w_1 . How to plot delta weight change?
#ax4.plot(t_array, w1_mean, color='c', label='mean')
# no fill between needed?


ax2.set_ylabel('weight_{ij}')
ax2.set_xlabel('t/tau_p')
ax2.grid()
ax2.legend()

ax3.set_ylabel('Bias')
ax3.set_xlabel('t/tau_p')
ax3.grid()
ax3.legend()

#ax4.set_ylabel('weight')
#ax3.set_xlabel('time t')
#ax3.grid()
#ax3.legend()

plt.show() #creates new plot window
# I can now use ax1, etc for new plots after plt.show()



# 0 presynaptic, 1 postsynaptic

# THIS IS APPARENTLY NOT WHAT I AM SUPPOSED TO BE DOING..-----

'''

def spike_timing_func(train_0, train_1):
    
    This function sorts the lists train_0 and train_1 of exact spike times of presynaptic and 
    postsynaptic spiking respectively, we sort them into one list together in the order in which 
    they appear (=spike), meaning the order of their spike timing. This is put into the list spike_times 
    and the corresponding index (0 for pre and 1 for post) is put into the list spike_indices. 
    
    spike_indices = []
    spike_times = []
    i = 0
    j = 0
    while i < len(train_0) or j < len(train_1): 
        # as long as there is element left in any of the two spike time lists
        i_done = i >= len(train_0) 
        j_done = j >= len(train_1)
        if not i_done: # as long as i < len(train_0)
                if j_done or train_0[i] <= train_1[j]: 
                    spike_indices.append(0)
                    spike_times.append(train_0[i])
                    i+=1
        if not j_done: # as long as j < len(train_1)
                if i_done or train_1[j] <= train_0[i]:
                    spike_indices.append(1)
                    spike_times.append(train_1[j])
                    j+=1
    return spike_indices, spike_times



def ltp_ltd_func(spike_indices, spike_times): #spike_indices = list of spike indices
    
    This function takes the spike_indices and spike_times lists from above and calculates the 
    lists of time intervals and of weight changes between every instance where we have either
    Pre-before-post (0 followed by 1) or Post-before-pre (1 followed by 0). 

    Since the STDP plot has negative spike timing intervals for Post-before-pre (left hand side)
    and positive spike timing intervals for Pre-before-post (right hand side), we calculate the
    spike timing intervals as [spike time of post-synaptic spike] - [spike time of pre-synaptic spike].
    Meaning, time intervals are time(post) - time(pre).
    
    time_intervals = [] #time intervals list
    weight_changes= [] #weight change list, will be returned in (%)
    print("len(spike_indices):", len(spike_indices))
    for i in range(1, len(spike_indices)-1200): #-1 normally
        if spike_indices[i] != spike_indices[i+1]: #if two following elem not the same, we are only interested in cases 0,1 or 1,0 
            pre_ind = i if spike_indices[i]==0 else i+1  #index of presynaptic neuron
            print('index of pre:' , pre_ind)
            post_ind = i if spike_indices[i]==1 else i+1 #index of postsynaptic neuron
            print('index of post:' , post_ind)
            time_intervals.append(spike_times[post_ind] - spike_times[pre_ind]) #time intervals given by time of post minus time of pre
            print('time interval: ', (spike_times[post_ind] - spike_times[pre_ind]))
            # because post-pre spiking should give negative value and pre-post positive value (look at picture)
            first_w = w_array[n_iterations-1][int(spike_times[i]/dt)] #the weight at specific spike time of pre divided by timestep dt 
            print('w(pre):', first_w)
            second_w = w_array[n_iterations-1][int(spike_times[i+1]/dt)] #the weight at specific spike time of post divided by timestep dt 
            print('w(post):', second_w)
            #wchange = (100 + (((post_w - pre_w)/pre_w) * 100))
            wchange = (((second_w - first_w)/first_w) * 100)
            print('w_change:', wchange)
            weight_changes.append(wchange)
    return time_intervals, weight_changes
     

spike_trains = spikemon.spike_trains()
train_0 = spike_trains[0] # exact spike times of presyn
train_1 = spike_trains[1] # exact spike times of postsyn
spike_indices, spike_times = spike_timing_func(train_0, train_1)
t_int_list, wchange_list = ltp_ltd_func(spike_indices, spike_times)

#print('first list: ', spike_indices) #returns list of 0s and 1s, the pre and post spiking. 
#print('second list', spike_times)

#plt.xlim(-15, 15)
#plt.ylim(-1.0, 1.0)
plt.xlabel('Spike timing interval')
plt.ylabel('Change in synaptic weight')
plt.legend
plt.scatter(t_int_list/ms, wchange_list, alpha=0.5) #alpha changes the transparency 
plt.grid()
plt.show()
'''



#-TODO
# plot these together - see if as in STDP
# clean up code & comment 
# split into pos and neg lists to make them different colors 



# how to get the weights out of these? hmmm how to measure them, how to actually simulate these
# ahh with ColumnCoords as above? And Then StimProtocol!! Look at Ferdinand's GitHub!!


# Choose total stimulation  time
# In Sim 1 t_total = 500 * ms #total simulation time.

# need this ? 

model = TullyNetwork() #Everything defined. Subclass of CorticalNetwork.
namespace = model.namespace # using defined attribute namespace from the TullyNetwork. 
# namespace used in Brian as dictionary of defined parameters and values. 

# STIMULI
# ColumnCoords, StimProtocol, StimTime all taken from stim_utils file.

# neuron_1_coords = ColumnCoords(0, 0) # first neuron, pre-synaptic.
# neuron_2_coords = ColumnCoords(1, 0) # second neuron, post-synaptic.
figure_2_stims = [
    StimProtocol(neuron_1_coords, StimTime(t_start=40.00*ms, t_end=40.01*ms)),   #pre-syn neuron 0-100ms
    #StimProtocol(neuron_2_coords, StimTime(t_start=100*ms, t_end=200*ms)), #post-syn neuron 100-200ms
]



# Difference between stimulus and spike.. How should I give it a spike at specific times ? 
# I try giving it different much stimulation and seeing how much and when it spikes and then adapt?
# Only want one presynaptic spike followed by one postsynaptic spike. 

# monitors

spikemon2 = model.add_spikemon()
weightmon = model.add_synmon(variables=['w'], record=True)

model.run(t_total)

# will need to plot weight after, but for now only want to plot spikemon

# from tully sim 1: composite.plot_traces(0, 1, spikemon, tracemon, syn_tracemon, model.S_REC, t_div=ms)

#spikemon = SpikeMonitor(G)

#run(50*ms)

print('Spike times: %s' % spikemon2.t[:])

#fig, (ax4) = plt.subplots(1, 1, sharex=True)
# måste ändra här så det mäts från rätt stimulering och ej förra
# is the spikemon same as before the problem?
# I think the spike trains are the problem ..
# normally spikemonitor spikemon = SpikeMonitor(G) takes group to record from (G here)
# how is this defined in model? aka  model = TullyNetwork() (subclass of Cortical Network)

#trains.compare_two_trains(ax4, spikemon2, 0, 1, t_div=NEW_TAU_P)

# do not actually need to use compare_two_trains, can also just plot both individually..
# look at documentation. 
# when gotten spike monitor to work, then start measuring weight and also do so it only spikes once
# if this is possible..

#plt.show()