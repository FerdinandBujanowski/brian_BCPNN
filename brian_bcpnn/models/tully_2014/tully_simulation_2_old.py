
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