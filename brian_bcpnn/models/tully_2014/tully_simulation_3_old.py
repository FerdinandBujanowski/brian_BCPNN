
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