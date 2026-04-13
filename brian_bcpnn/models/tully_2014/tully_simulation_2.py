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
w1_array = np.zeros(shape=(n_iterations, int(t_total/defaultclock.dt)))
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
    if t_array is None:
        t_array = weightmon.t/model.namespace['tau_p'] #x-axis: time divided by tau_p

#  w1_array[i,:] = weightmon.w[0]
#    if t1_array is None:
#        t1_array = weightmon.t #plot only time on x axis and not divided by tau_p

    beta_array[i,:] = biasmon.beta[0]
    if beta_array is None:
        beta_array = biasmon.t/model.namespace['tau_p']
    

# calculate and plot stats for w and bias
# all I did for weight need to do for bias
# need to add third axis ax3 for bias and plot this 
# also I am only plotting bias for postsynaptic neuron 

w_mean = np.mean(w_array, axis=0)
w_std = np.std(w_array, axis=0)
n_std = 1.96

w1_mean = np.mean(w1_array, axis=0)

beta_mean = np.mean(beta_array, axis=0)
beta_std = np.std(beta_array, axis=0)

# ax1 is spikemonitor (?)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
trains.compare_two_trains(ax1, spikemon, 0, 1, t_div=NEW_TAU_P)

ax2.plot(t_array, w_mean, color='c', label='mean')
ax2.fill_between(t_array, w_mean-n_std*w_std, w_mean+n_std*w_std, color='c', alpha=0.3, label='95%')

ax3.plot(t_array, beta_mean, color='m', label='mean')
ax3.fill_between(t_array, beta_mean-n_std*beta_std, beta_mean+n_std*beta_std, color='m', alpha=0.3, label='95')

# do for ax4 delta weight change. so w_2 - w_1 . How to plot delta weight change?
ax4.plot(t_array, w1_mean, color='c', label='mean')
# no fill between needed?


ax2.set_ylabel('weight_{ij}')
ax2.set_xlabel('t/tau_p')
ax2.grid()
ax2.legend()

ax3.set_ylabel('Bias')
ax3.set_xlabel('t/tau_p')
ax3.grid()
ax3.legend()

ax4.set_ylabel('weight')
ax3.set_xlabel('time t')
ax3.grid()
ax3.legend()

plt.show() #creates new plot window
# I can now use ax1, etc for new plots after plt.show()

# zero presynaptic, one postsynaptic

def spike_timing_func(train_0, train_1):
    spike_indices = []
    spike_times = []
    i = 0
    j = 0
    while i < len(train_0) or j < len(train_1):
        i_done = i >= len(train_0)
        j_done = j >= len(train_1)
        if not i_done:
                if j_done or train_0[i] <= train_1[j]:
                    spike_indices.append(0)
                    spike_times.append(train_0[i])
                    i+=1
        if not j_done:
                if i_done or train_1[j] <= train_0[i]:
                    spike_indices.append(1)
                    spike_times.append(train_1[j])
                    j+=1
    return spike_indices, spike_times


#train_0, train_1 = trains.compare_two_trains(ax1, spikemon, 0, 1, t_div=NEW_TAU_P)
#print(train_0, train_1)
spike_trains = spikemon.spike_trains()
train_0 = spike_trains[0] # exact spike times presyn
train_1 = spike_trains[1] #post
s_indice, s_times = spike_timing_func(train_0, train_1)
print('first list: ', s_indice) #returns list of 0s and 1s, the pre and post spiking. 
#print('second list', s_times)


# we only want times for pre-post or post-pre

def ltp_ltd_func(l_ind, s_times):
    t_int_list = [] #time intervals list
    wchange_list = [] #weight change list 
    for i in range(len(l_ind)):
        if l_ind[i] != l_ind[i+1]: #if two following elem not the same
            #only interested in cases 0,1 or 1,0 
            pre_ind = i if l_ind[i]==0 else i+1  #index of presynaptic neuron
            post_ind = i if l_ind[i]==1 else i+1 #index of postsynaptic neuron
            t_int_list.append(s_times[post_ind] - s_times[pre_ind]) #time intervals given by time of post minus time of pre
            # because post-pre spiking should give negative value and pre-post positive value (look at picture)
            pre_w = weightmon[0][int(s_times[pre_ind]/dt)] #the weight at specific spike time of pre divided by timestep dt 
            post_w = weightmon[0][int(s_times[post_ind]/dt)] #same but for post

            # ARE EITHER OF THESE OR THE BELOW CORRECT?? WHAT IS THIS?

            wchange_pre = (pre_w[i+1] - pre_w[i])/pre_w[i] * 100
            wchange_post = (post_w[i+1] - post_w[i])/post_w[i] * 100

            wchange = ((post_w - pre_w)/pre_w) * 100 # i do not think this is correct
            wchange_list.append(wchange)


         #   (post_w - pre_w)/pre_w x 100 = 
         #   (pre_w - post_w)/post_w x 100 = 


# do I have to care about this now? 


ltp_ltd_func(s_indice, s_times)


# now plot the time differences (so neg and pos I guess?) and the weights from weightmon? 
# i do not remember what to do with the weight??

#divide by 0.1 (dt?) and take as integer 
